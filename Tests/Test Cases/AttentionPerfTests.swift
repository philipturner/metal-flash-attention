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
    print()
    
    // 8 for small, 128 for large, length 1 while prototyping
    // 4 for small, 64 for large, length 2 for production
    let duration = Duration(granularity: 64, length: 2)
    let (domain, ranges) = rangeSequenceScaling(
      duration: duration, isLarge: true)
    testAttention(
      domain: domain, ranges: ranges, backends: [.mps, .mfa])
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
    case mps = 0
    case mpsMasked = 1
    case mfa = 2
    case mfaTriangular = 3
    case mfaBlockSparse = 4
    
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
    
    var mplColor: String {
      switch self {
      case .mps: return "-r"
      case .mpsMasked: return "-y"
      case .mfa: return "-b"
      case .mfaTriangular: return "-g"
      case .mfaBlockSparse: return "-k"
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
  ) -> (ClosedRange<Int>, [AttentionRange]) {
    let granularity = duration.granularity
    
    var domain: ClosedRange<Int>
    var parameters: [SIMD4<Int>]
    if !isLarge {
      domain = 0...2048
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
      domain = 2048...8192
      parameters = [
        SIMD4(2048, 3072, 16, 8),
        SIMD4(3072, 4096,  8, 8),
        SIMD4(4096, 6144,  4, 8),
        SIMD4(6144, 8193,  2, 8),
      ]
    }
    
    let headCount = isLarge ? 5 : 10
    return (domain, parameters.map { parameter in
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
    })
  }
  
#if DEBUG
  static let verifyResults = true
#else
  static let verifyResults = false
#endif
  
  func testAttention(
    domain: ClosedRange<Int>,
    ranges: [AttentionRange],
    backends: [AttentionBackend]
  ) {
    struct Tensors {
      var q: Tensor<Real>
      var k: Tensor<Real>
      var v: Tensor<Real>
      var o: Tensor<Real>
      var mask: Tensor<Real>?
      
      init(config: AttentionConfig) {
        var B: [Int] = []
        if config.B > 1 {
          B += [config.B]
        }
        
        let R = config.R
        let C = config.C
        let H = config.H
        let D = config.D
        let shapeQ = B + [R, H, D]
        let shapeK = B + [C, H, D]
        let shapeV = B + [C, H, D]
        let shapeO = B + [R, H, D]
        self.q = Tensor(shape: shapeQ, randomUniform: 0..<1)
        self.k = Tensor(shape: shapeK, randomUniform: 0..<1)
        self.v = Tensor(shape: shapeV, randomUniform: 0..<1)
        self.o = Tensor(zerosLike: shapeO)
        self.mask = nil
      }
      
      init(_ other: Tensors, backend: TensorBackend) {
        self.q = Tensor(copying: other.q, backend: backend)
        self.k = Tensor(copying: other.k, backend: backend)
        self.v = Tensor(copying: other.v, backend: backend)
        self.o = Tensor(copying: other.o, backend: backend)
        if let mask = other.mask {
          self.mask = Tensor(copying: mask, backend: backend)
        }
      }
    }
    
    @discardableResult
    func runAttention(
      config: AttentionConfig,
      trials: Int,
      iterations: Int
    ) -> Float {
      var tensors = Tensors(config: config)
      var minTime: Double = .infinity
      let backend = TensorBackend.default
      for _ in 0..<trials {
        backend.markFirstCommand()
        for _ in 0..<iterations {
          tensors.o.attention(
            queries: tensors.q,
            keys: tensors.k,
            values: tensors.v,
            transposeK: true)
        }
        backend.markLastCommand()
        minTime = min(minTime, backend.synchronize())
      }
      
      var floatOps = config.B
      floatOps *= config.R
      floatOps *= config.C
      floatOps *= config.H
      floatOps *= (4 * config.D + 10)
      floatOps *= iterations
      
      let flops = Double(floatOps) / minTime
      return Float(flops / 1e9)
    }
    
    func verifyResults(
      config: AttentionConfig,
      backendMPS: AttentionBackend
    ) {
      var mfaTensors = Tensors(config: config)
      let mps = backendMPS.tensorBackend
      var mpsTensors = Tensors(mfaTensors, backend: mps)
      
      _ExecutionContext.profileCommands {
        mfaTensors.o.attention(
          queries: mfaTensors.q,
          keys: mfaTensors.k,
          values: mfaTensors.v,
          transposeK: true)
      }
      
      _ExecutionContext.withDefaultBackend(mps) {
        _ExecutionContext.profileCommands {
          mpsTensors.o.attention(
            queries: mpsTensors.q,
            keys: mpsTensors.k,
            values: mpsTensors.v,
            transposeK: true)
        }
      }
      
      let params = EuclideanDistanceParameters(
        averageMagnitude: 1.0,
        averageDeviation: 0.2,
        batchSize: config.B * config.H)
      let failed = !mfaTensors.o.isApproximatelyEqual(
        to: mpsTensors.o, parameters: params)
      if failed {
        print("Failure: \(config.description)")
        
        let dist = mfaTensors.o.euclideanDistance(to: mpsTensors.o)
        print(" - Distance: \(String(format: "%.3f", dist))")
        print(" - Cannot visualize because O_trans not true.")
      }
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
              
              if record {
                samples[index] = (config, gflops)
              }
            }
          }
          
          for (config, gflops) in samples {
            let backendRepr = backend.description
            let configRepr = config.description
            let gflopsRepr = Int(round(gflops))
            print("(\(backendRepr)) \(configRepr) - \(gflopsRepr) GFLOPS")
          }
          
          data.append(backend: backend, data: samples)
          progresses[backend] = range
        }
      }
    }
    
    let H = ranges.first!.exclusiveEnd.H
    let D = ranges.first!.exclusiveEnd.D
    let title = "H=\(H), D=\(D)"
    graph(
      data: data,
      axis: \.R,
      domain: domain,
      independentVariable: "Sequence Length",
      title: title)
  }
  
  func graph(
    data: AttentionData,
    axis: KeyPath<AttentionConfig, Int>,
    domain: ClosedRange<Int>,
    independentVariable: String,
    title: String
  ) {
    let plt = PythonContext.global.plt
    for key in data.gflops.keys.sorted(by: { $0.rawValue < $1.rawValue }) {
      let value = data.gflops[key]!
      let label = key.description
      let sizeArray = value.map { $0.0[keyPath: axis] }
      
      let gflopsArray = value.map { $0.1 }
      let style = key.mplColor
      plt.plot(sizeArray, gflopsArray, style, label: label)
    }
    plt.legend(loc: "upper left")
    plt.xlim(domain.lowerBound, domain.upperBound)
    plt.ylim(0, MetalContext.global.infoDevice.flops / 1e9)
    plt.xlabel(independentVariable)
    plt.ylabel("GFLOPS")
    
#if DEBUG
    let debugWarning = " (NOT USABLE FOR CI)"
#else
    let debugWarning = ""
#endif
    if Real.self == Float.self {
      plt.title("FlashAttention (F32, \(title))\(debugWarning)")
    } else {
      plt.title("FlashAttention (F16, \(title))\(debugWarning)")
    }
    plt.show()
  }
}
