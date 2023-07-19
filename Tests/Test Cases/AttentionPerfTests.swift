//
//  AttentionPerfTests.swift
//  MetalFlashAttention
//
//  Created by Philip Turner on 7/18/23.
//

import Metal
import QuartzCore

class AttentionPerfTests: MFATestCase {
  override class func typeDescription() -> String {
    "AttentionPerfTests"
  }
  
  override func runVeryLongTests() {
    let duration = Duration(granularity: 2, length: 1)
    let ranges = rangeSequenceScaling(duration: duration, isLarge: false)
    _ = testAttention(ranges: ranges, backends: [.mps, .mfa])
  }
  
  struct AttentionConfig: Equatable, Hashable {
    var B: Int
    var R: Int
    var C: Int
    var H: Int
    var D: Int
    var sparsityPercent: Int
    
    var description: String {
      var output = "\(R)x\(C)x\(H)x\(D)"
      if B > 1 {
        output = "\(B)x\(output)"
      }
      if sparsityPercent > 0 && sparsityPercent < 100 {
        output = "\(sparsityPercent)% \(output)"
      }
      return output
    }
    
    static func + (
      lhs: AttentionConfig,
      rhs: AttentionConfig
    ) -> AttentionConfig {
      var out = lhs
      out.B += rhs.B
      out.R += rhs.R
      out.C += rhs.C
      out.H += rhs.H
      out.D += rhs.D
      out.sparsityPercent += rhs.sparsityPercent
      return out
    }
    
    func less(than rhs: AttentionConfig, axes: AttentionConfig) -> Bool {
      if axes.B > 0 {
        if self.B < rhs.B {
          return true
        }
      }
      if axes.R > 0 {
        if self.R < rhs.R {
          return true
        }
      }
      if axes.C > 0 {
        if self.C < rhs.C {
          return true
        }
      }
      if axes.H > 0 {
        if self.H < rhs.H {
          return true
        }
      }
      if axes.D > 0 {
        if self.D < rhs.D {
          return true
        }
      }
      if axes.sparsityPercent > 0 {
        if self.sparsityPercent < rhs.sparsityPercent {
          return true
        }
      }
      return false
    }
  }
  
  enum AttentionBackend: Int {
    case mps
    case mpsMasked
    case mfa
    case mfaTriangular
    case mfaBlockSparse
    
    var tensorBackend: TensorBackend {
      switch self {
      case .mps: return .mps
      case .mfa: return .mfa
      default: fatalError("Attention backend not supported yet.")
      }
    }
    
    var description: String {
      switch self {
      case .mps: return "MPS"
      case .mfa: return "MFA"
      default: fatalError("Attention backend not supported yet.")
      }
    }
    
    var isMPS: Bool {
      switch self {
      case .mps: return true
      case .mfa: return false
      default: fatalError("Attention backend not supported yet.")
      }
    }
    
    var equivalentMPS: AttentionBackend {
      switch self {
      case .mps: fatalError("Is MPS.")
      case .mpsMasked: fatalError("Is MPS.")
      case .mfa: return .mps
      case .mfaTriangular: return .mpsMasked
      case .mfaBlockSparse: return .mpsMasked
      }
    }
    
    var isTriangularMasked: Bool {
      switch self {
      case .mps: return false
      case .mfa: return false
      default: return true
      }
    }
    
    var isBlockSparse: Bool {
      switch self {
      case .mfaBlockSparse: return true
      default: return false
      }
    }
  }
  
  struct AttentionData {
    var gflops: [AttentionBackend: [(AttentionConfig, Float)]]
    
    init(backends: [AttentionBackend]) {
      gflops = [:]
      for backend in backends {
        gflops[backend] = []
      }
    }
    
    mutating func append(
      backend: AttentionBackend,
      data: [(AttentionConfig, Float)]
    ) {
      self.gflops[backend]!.append(contentsOf: data)
    }
    
    static func + (
      lhs: AttentionData,
      rhs: AttentionData
    ) -> AttentionData {
      var out = lhs
      for backend in lhs.gflops.keys {
        out.gflops[backend]! += rhs.gflops[backend]!
      }
      return out
    }
  }
  
  struct AttentionRange: Equatable, Hashable {
    var cursor: AttentionConfig
    var exclusiveEnd: AttentionConfig
    var stride: AttentionConfig
    var unfinished: Bool {
      cursor.less(than: exclusiveEnd, axes: stride)
    }
    
    // Duplicate the first test (all trials), discard the sample.
    // Then duplicate the last test (all trials), discard the sample.
    // Finally, start the trials.
    var testsPerContextSwitch: Int
    var commandsPerEncoder: Int
    var trials: Int
    var commandsPerEncoderMPS: Int
    var trialsMPS: Int
    
    func getIterations(backend: AttentionBackend) -> Int {
      if backend.isMPS {
        return commandsPerEncoderMPS
      } else {
        return commandsPerEncoder
      }
    }
    
    func getTrials(backend: AttentionBackend) -> Int {
      if backend.isMPS {
        return trialsMPS
      } else {
        return trials
      }
    }
  }
  
  struct Duration {
    var granularity: Int
    var length: Int
  }
  
  func rangeSequenceScaling(
    duration: Duration, isLarge: Bool
  ) -> [AttentionRange] {
    let granularity = duration.granularity
    
    var parameters: [SIMD4<Int>]
    if !isLarge {
      precondition(duration.granularity == 2)
      parameters = [
        SIMD4(granularity, 192, 256, 8),
        SIMD4( 192,  256, 128, 8),
        SIMD4( 256,  384,  64, 8),
        SIMD4( 384,  512,  32, 8),
        SIMD4( 512,  768,  16, 8),
        SIMD4( 768, 1024,   8, 8),
        SIMD4(1024, 1536,   4, 8),
        SIMD4(1536, 2049,   2, 8),
      ]
    } else {
      precondition(granularity >= 16, "Granularity is too small.")
      parameters = [
        SIMD4(2048, 3072, 16, 8),
        SIMD4(3072, 4096,  8, 8),
        SIMD4(4096, 6144,  4, 8),
        SIMD4(6144, 8192,  2, 8),
      ]
    }
    
    let headCount = isLarge ? 5 : 10
    return parameters.map { parameter in
      let start = AttentionConfig(
        B: 1,
        R: parameter[0],
        C: parameter[0],
        H: headCount,
        D: 64,
        sparsityPercent: -1)
      let end = AttentionConfig(
        B: 1,
        R: parameter[1],
        C: parameter[1],
        H: headCount,
        D: 64,
        sparsityPercent: -1)
      let stride = AttentionConfig(
        B: 0,
        R: granularity,
        C: granularity,
        H: 0,
        D: 0,
        sparsityPercent: 0)
      
      var iterations = max(1, parameter[2])
      var trials = 0
      let ref = parameter[3] * duration.length
      SquareMatrixBenchmark_configure_2(
        &iterations, &trials, ref: ref)
      
      var iterationsMPS = max(1, min(32, parameter[2]))
      var trialsMPS = 0
      let refMPS = parameter[3]
      SquareMatrixBenchmark_configure_2(
        &iterationsMPS, &trialsMPS, ref: refMPS)
      
      return AttentionRange(
        cursor: start,
        exclusiveEnd: end,
        stride: stride,
        testsPerContextSwitch: 16,
        commandsPerEncoder: iterations,
        trials: trials,
        commandsPerEncoderMPS: iterationsMPS,
        trialsMPS: trialsMPS)
    }
  }
  
#if DEBUG
static let verifyResults = true
#else
static let verifyResults = false
#endif
  
  func testAttention(
    ranges: [AttentionRange],
    backends: [AttentionBackend]
  ) -> AttentionData {
    
    @discardableResult
    func runAttention(
      config: AttentionConfig,
      trials: Int,
      iterations: Int
    ) -> Float {
      let backend = TensorBackend.default
      backend.markFirstCommand()
      backend.markLastCommand()
      _ = backend.synchronize()
      return 0
    }
    
    func verifyResults(
      config: AttentionConfig,
      backendMPS: AttentionBackend
    ) {
      
    }
    
    var data = AttentionData(backends: backends)
    for range in ranges {
      var progresses: [AttentionBackend: AttentionRange] = [:]
      for backend in backends {
        progresses[backend] = range
      }
      
      while progresses.values.contains(where: \.unfinished) {
        for backend in backends {
          var range = progresses[backend]!
          var samples: [(AttentionConfig, Float)] = []
          
          let tensorBackend = backend.tensorBackend
          _ExecutionContext.withDefaultBackend(tensorBackend) {
            tensorBackend.withGhostExecution {
              for _ in 0..<range.testsPerContextSwitch {
                guard range.unfinished else {
                  break
                }
                samples.append((range.cursor, 0))
                runAttention(
                  config: range.cursor,
                  trials: 1,
                  iterations: 1)
                range.cursor = range.cursor + range.stride
              }
            }
            if Self.verifyResults, !backend.isMPS {
              for (config, _) in samples {
                verifyResults(
                  config: config,
                  backendMPS: backend.equivalentMPS)
                print("Verified results for config \(config)")
              }
            }
            
            var iterations: [(Int, Bool)] = [
              (0, false),
              (samples.count - 1, false)
            ]
            for index in samples.indices {
              iterations.append((index, true))
            }
            
            for (index, record) in iterations {
              let config = samples[index].0
              let gflops = runAttention(
                config: config,
                trials: range.getTrials(backend: backend),
                iterations: range.getIterations(backend: backend))
              
              // TODO: Allow certain ranges to be blacklisted for MPS, so we can
              // represent "error" as 0 GFLOPS.
              if record {
                samples[index] = (config, gflops)
              }
            }
          }
          
          // TODO: Dry run, just adding samples to the console.
          for (config, gflops) in samples {
            let backendRepr = backend.description
            let configRepr = config.description
            let gflopsRepr = Int(round(gflops))
            print("(\(backendRepr)) \(configRepr) - \(gflopsRepr)")
          }
          
          data.append(backend: backend, data: samples)
          progresses[backend] = range
        }
      }
    }
    
    return data
  }
  
  // TODO: Function to graph data in Matplotlib along one axis.
}
