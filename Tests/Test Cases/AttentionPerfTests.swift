//
//  AttentionPerfTests.swift
//  MetalFlashAttention
//
//  Created by Philip Turner on 7/18/23.
//

import Metal
import PythonKit
import QuartzCore

class AttentionPerfTests: MFATestCase {
  override class func typeDescription() -> String {
    "AttentionPerfTests"
  }
  
  override func runVeryLongTests() {
    print()
    
    // Prototyping:
    // Granularity:
    //   4 -> 1 (causal), 8 (small), 128 (large), 1+ (head size)
    // Length:
    //   2 (causal), 1 (everything else)
    // For causal:
    //   remove the non-MFA backends
    //   narrow the range from 0...1024 to 512...1024
    // For heads scaling:
    //   sequence length 2048
    //
    // Production:
    // Granularity:
    //   2 (causal), 4 (small), 64 (large), 1+ (head size)
    // Length:
    //   1 (head size), 2 (everything else)
    // For heads scaling:
    //   sequence length 4096
    
    let duration = Duration(granularity: 4, length: 2)
    let (domain, ranges) = rangeSequenceScaling(
      duration: duration, type: .causal)

    var backends = SequenceType.causal.backends
//    let backends: [AttentionBackend] = [.mfa]
    
    backends = backends.compactMap {
      if $0.isMPS { return nil }
      return $0
    }
    
//    let duration = Duration(granularity: 1, length: 1)
//    let (domain, ranges) = rangeHeadScaling(duration: duration)
//    let backends = [AttentionBackend.mps, AttentionBackend.mfa]
    testAttention(
      domain: domain, ranges: ranges, backends: backends)
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
    case mfaMasked = 3
    case mfaBlockSparse = 4
    
    var tensorBackend: TensorBackend {
      switch self {
      case .mps, .mpsMasked: return .mps
      case .mfa, .mfaMasked, .mfaBlockSparse: return .mfa
      }
    }
    
    var causalDescription: String {
      switch self {
      case .mps: return "MPS"
      case .mpsMasked: return "MPS (Causal)"
      case .mfa: return "MFA"
      case .mfaMasked: return "MFA (Causal, Dense)"
      case .mfaBlockSparse: return "MFA (Causal, Sparse)"
      }
    }
    
    var isMPS: Bool {
      switch self {
      case .mps, .mpsMasked: return true
      case .mfa, .mfaMasked, .mfaBlockSparse: return false
      }
    }
    
    var equivalentMPS: AttentionBackend {
      switch self {
      case .mps: fatalError("Is MPS.")
      case .mpsMasked: fatalError("Is MPS.")
      case .mfa: return .mps
      case .mfaMasked: return .mpsMasked
      case .mfaBlockSparse: return .mpsMasked
      }
    }
    
    var isMasked: Bool {
      switch self {
      case .mps, .mfa: return false
      case .mpsMasked, .mfaMasked, .mfaBlockSparse: return true
      }
    }
    
    var mplColor: String {
      switch self {
      case .mps: return "-r"
      case .mpsMasked: return "-y"
      case .mfa: return "-b"
      case .mfaMasked: return "-g"
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
  
  enum SequenceType {
    case small
    case large
    case causal
    
    var backends: [AttentionBackend] {
      switch self {
      case .small, .large:
        return [.mps, .mfa]
      case .causal:
        return [.mps, .mpsMasked, .mfa, .mfaMasked, .mfaBlockSparse]
      }
    }
  }
  
  func rangeSequenceScaling(
    duration: Duration, type: SequenceType
  ) -> (ClosedRange<Int>, [AttentionRange]) {
    let granularity = duration.granularity
    
    var domain: ClosedRange<Int>
    var parameters: [SIMD4<Int>]
    if type == .causal {
      domain = 512...1024
//      domain = 0...1024
      parameters = [
//        SIMD4(granularity, 192, 256, 8),
//        SIMD4( 192,  256, 128, 8),
//        SIMD4( 256,  384,  64, 8),
//        SIMD4( 384,  512,  32, 8),
        SIMD4( 512,  768,  16, 8),
        SIMD4( 768, 1025,   8, 8),
      ]
    } else if type == .small {
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
    
    var headCount: Int
    if type == .large {
      headCount = 5
    } else {
      headCount = 10
    }
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
  
  func rangeHeadScaling(
    duration: Duration
  ) -> (ClosedRange<Int>, [AttentionRange]) {
    let granularity = duration.granularity
    let domain = 0...384
    let parameters: [SIMD4<Int>] = [
      // Prototyping:
//      SIMD4(granularity, 32,  16, granularity * 1),
//      SIMD4(         32, 64,   8, granularity * 1),
//      SIMD4(         64, 160,  4, granularity * 2),
//      SIMD4(        160, 385,  4, granularity * 4),
      
      // Production:
      SIMD4(granularity, 32,  16, granularity * 1),
      SIMD4(         32, 64,   8, granularity * 1),
      SIMD4(         64, 128,  4, granularity * 2),
      SIMD4(        128, 256,  2, granularity * 4),
      SIMD4(        256, 385,  2, granularity * 16),
    ]
    let sequenceLength = 4096
    
    return (domain, parameters.indices.map { i in
      let parameter = parameters[i]
      let start = AttentionConfig(
        B: 1,
        R: sequenceLength,
        C: sequenceLength,
        H: 8,
        D: parameter[0],
        sparsityPercent: -1)
      let end = AttentionConfig(
        B: 1,
        R: sequenceLength,
        C: sequenceLength,
        H: 8,
        D: parameter[1],
        sparsityPercent: -1)
      let stride = AttentionConfig(
        B: 0,
        R: 0,
        C: 0,
        H: 0,
        D: parameter[3],
        sparsityPercent: -1)
      
      var iterations = max(1, parameter[2])
      var trials = 0
      let ref = 8 * duration.length
      SquareMatrixBenchmark_configure_2(
        &iterations, &trials, ref: ref)
      
      var iterationsMPS = max(1, min(32, parameter[2]))
      var trialsMPS = 0
      let refMPS = 8
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
    backends _backends: [AttentionBackend]
  ) {
    let backends = _backends.sorted(by: {
      $0.rawValue < $1.rawValue
    })
    
    struct Tensors {
      var q: Tensor<Real>
      var k: Tensor<Real>
      var v: Tensor<Real>
      var o: Tensor<Real>
      var mask: Tensor<Real>?
      
      init(config: AttentionConfig, backend: AttentionBackend) {
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
        
        if backend.isMasked {
          let shapeMask = B + [1, R, C]
          self.mask = Tensor(shape: shapeMask, mask: .upperTriangular)
        }
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
      backend: AttentionBackend,
      trials: Int,
      iterations: Int
    ) -> Float {
      var tensors = Tensors(config: config, backend: backend)
      var minTime: Double = .infinity
      let tensorBackend = TensorBackend.default
      for _ in 0..<trials {
        tensorBackend.markFirstCommand()
        for _ in 0..<iterations {
          tensors.o.attention(
            queries: tensors.q,
            keys: tensors.k,
            values: tensors.v,
            mask: tensors.mask,
            transposeK: true,
            blockSparse: backend == .mfaBlockSparse)
        }
        tensorBackend.markLastCommand()
        minTime = min(minTime, tensorBackend.synchronize())
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
      backend: AttentionBackend
    ) {
      precondition(!backend.isMPS)
      var mfaTensors = Tensors(config: config, backend: backend)
      let mps = backend.equivalentMPS.tensorBackend
      var mpsTensors = Tensors(mfaTensors, backend: mps)
      
      _ExecutionContext.profileCommands {
        mfaTensors.o.attention(
          queries: mfaTensors.q,
          keys: mfaTensors.k,
          values: mfaTensors.v,
          mask: mfaTensors.mask,
          transposeK: true,
          blockSparse: backend == .mfaBlockSparse)
      }
      
      _ExecutionContext.withDefaultBackend(mps) {
        _ExecutionContext.profileCommands {
          mpsTensors.o.attention(
            queries: mpsTensors.q,
            keys: mpsTensors.k,
            values: mpsTensors.v,
            mask: mpsTensors.mask,
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
                  backend: backend,
                  trials: 1,
                  iterations: 1)
                range.cursor = range.cursor + range.stride
              }
              if range.cursor.less(
                than: range.exclusiveEnd, axes: range.stride
              ) {
                if !(range.cursor + range.stride).less(
                  than: range.exclusiveEnd, axes: range.stride
                ) {
                  // Add one more sample.
                  samples.append((range.cursor, 0))
                  runAttention(
                    config: range.cursor,
                    backend: backend,
                    trials: 1,
                    iterations: 1)
                  range.cursor = range.cursor + range.stride
                }
              }
            }
            if Self.verifyResults, !backend.isMPS {
              for (config, _) in samples {
                verifyResults(
                  config: config,
                  backend: backend)
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
                backend: backend,
                trials: range.getTrials(backend: backend),
                iterations: range.getIterations(backend: backend))
              
              if record {
                samples[index] = (config, gflops)
              }
            }
          }
          
          for (config, gflops) in samples {
            let backendRepr = backend.causalDescription
            let configRepr = config.description
            let gflopsInt = Int(round(gflops))
            print("(\(backendRepr)) \(configRepr) - \(gflopsInt) GFLOPS")
            
            if backendRepr.contains("MFA") {
//              print("\(config.D), \(gflopsInt)")
//              print("\(config.R), \(gflopsInt)")
            }
          }
          
          data.append(backend: backend, data: samples)
          progresses[backend] = range
        }
      }
    }
    
    let stride = ranges.first!.stride
    if stride.R > 0, stride.C > 0, stride.H == 0, stride.D == 0 {
      let H = ranges.first!.exclusiveEnd.H
      let D = ranges.first!.exclusiveEnd.D
      let title = "H=\(H), D=\(D)"
      graph(
        data: data,
        axis: \.R,
        domain: domain,
        independentVariable: "Sequence Length",
        logThreshold: nil,
        title: title)
    } else if stride.R == 0 && stride.C == 0, stride.H == 0, stride.D > 0 {
      let R = ranges.first!.exclusiveEnd.R
      let H = ranges.first!.exclusiveEnd.H
      let title = "R=C=\(R), H=\(H)"
      graph(
        data: data,
        axis: \.D,
        domain: domain,
        independentVariable: "Head Size",
        logThreshold: 16,
        title: title)
    } else {
      fatalError("Unsupported configuration")
    }
  }
  
  func graph(
    data: AttentionData,
    axis: KeyPath<AttentionConfig, Int>,
    domain: ClosedRange<Int>,
    independentVariable: String,
    logThreshold: Int?,
    title: String
  ) {
    let plt = PythonContext.global.plt
    let (_, ax) = plt.subplots().tuple2
    
    var maxListGFLOPS: Float = 0
    for key in data.gflops.keys.sorted(by: { $0.rawValue < $1.rawValue }) {
      let value = data.gflops[key]!
      let label = key.causalDescription
      let sizeArray = value.map { $0.0[keyPath: axis] }
      
      let gflopsArray = value.map { $0.1 }
      let style = key.mplColor
      plt.plot(sizeArray, gflopsArray, style, label: label)
      maxListGFLOPS = max(maxListGFLOPS, gflopsArray.max()!)
    }
    plt.legend(loc: "upper left")
    plt.xlim(domain.lowerBound, domain.upperBound)
    
    var maxGFLOPS = Float(MetalContext.global.infoDevice.flops / 1e9)
    if maxListGFLOPS > maxGFLOPS {
      let xPoints: [Int] = [domain.lowerBound, domain.upperBound]
      let yPoints: [Float] = [maxGFLOPS, maxGFLOPS]
      plt.plot(xPoints, yPoints, linestyle: "dashed", color: "black")
      
      maxGFLOPS = maxListGFLOPS
    }
    
    plt.ylim(0, maxGFLOPS)
    plt.xlabel(independentVariable)
    
    if axis == \AttentionConfig.sparsityPercent {
      plt.ylabel("GFLOPS * sparsity")
    } else {
      plt.ylabel("GFLOPS")
    }
    
    if let logThreshold {
      plt.xscale("symlog", base: 2, linthresh: logThreshold)
      
      let ticker = Python.import("matplotlib.ticker")
      ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
      plt.xticks([1, 8, 16, 32, 64, 128, 256, 384])
    }
    
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
