//
//  PerformanceTests.swift
//  MetalFlashAttention
//
//  Created by Philip Turner on 6/27/23.
//

import Metal
import QuartzCore

class PerformanceTests: MFATestCase {
  override class func typeDescription() -> String {
    "PerformanceTests"
  }
  
  override func runVeryLongTests() {
    // Tests the precision you set as the global testing precision. For a quick
    // smoke test, you can set a larger granularity.
    testGEMMSpeed(
      granularity: 2, trialsExtension: 2, B_trans: true, batchSize: 2)
    
//    testGEMMSpeed(
//      granularity: 16, trialsExtension: 1, B_trans: true, batchSize: 2)
  }
  
  // Covers the entire range of square matrix sizes, as well as differences
  // between MFA 32x32, MFA 48x48, and MPS.
  func testGEMMSpeed(
    granularity: Int,
    trialsExtension: Int,
    A_trans: Bool = false,
    B_trans: Bool = false,
    batchSize: Int? = nil
  ) {
    precondition(granularity.nonzeroBitCount == 1)
    let logProgress = true
    
    enum Config: CaseIterable {
      // Ordered from fastest to slowest at large matrix sizes.
      case mfa48x48
      case mfa32x32
      case mps
      
      static var fastConfigs: [Config] { [.mfa48x48, .mfa32x32, .mps] }
      
      var backend: TensorBackend {
        if self == .mps { return .mps }
        else { return .mfa }
      }
      
      var name: String {
        switch self {
        case .mfa48x48: return "MFA 48x48"
        case .mfa32x32: return "MFA 32x32"
        case .mps: return "MPS"
        }
      }
      
      func prepare() {
        _ExecutionContext.defaultBackend = self.backend
        if self == .mfa32x32 {
          let M_simd = MFA_GEMM.functionConstants["M_simd"] as! UInt16
          let N_simd = MFA_GEMM.functionConstants["N_simd"] as! UInt16
          let K_simd = MFA_GEMM.functionConstants["K_simd"] as! UInt16
          precondition(M_simd == 16)
          precondition(N_simd == 16)
          precondition(K_simd == 32)
        }
        if self == .mfa48x48 {
          MFA_Backend.global.cache.clear()
          MFA_GEMM.functionConstants["M_simd"] = UInt16(24)
          MFA_GEMM.functionConstants["N_simd"] = UInt16(24)
          let K_simd = (Real.self == Float.self) ? 24 : 32
          MFA_GEMM.functionConstants["K_simd"] = UInt16(K_simd)
        }
      }
      
      func cleanup() {
        _ExecutionContext.defaultBackend = .numpy
        if self == .mfa48x48 {
          MFA_Backend.global.cache.clear()
          MFA_GEMM.functionConstants["M_simd"] = UInt16(16)
          MFA_GEMM.functionConstants["N_simd"] = UInt16(16)
          MFA_GEMM.functionConstants["K_simd"] = UInt16(32)
        }
      }
    }
    
    struct Segment {
      var sizes: Range<Int>
      var iterations: Int
      var flops: [Config: [Double]] = [:]
      var currentConfig: Config?
      
      init(sizes: Range<Int>, iterations: Int) {
        self.sizes = sizes
        self.iterations = iterations
        flops[.mfa48x48] = []
        flops[.mfa32x32] = []
        flops[.mps] = []
      }
      
      mutating func prepare(config: Config) {
        self.currentConfig = config
        config.prepare()
      }
      
      mutating func cleanup(config: Config) {
        config.cleanup()
        self.currentConfig = nil
      }
      
      // If initial, this will run a ghost pass.
      mutating func _profile(
        sizes: Range<Int>, granularity: Int,
        trialsExtension: Int, isInitial: Bool,
        A_trans: Bool, B_trans: Bool, batchSize: Int?
      ) {
        func innerLoop(size: Int, reportResults: Bool) {
          if size % granularity != 0 {
            if !isInitial && reportResults {
              self.flops[currentConfig!]!.append(0)
            }
            return
          }
          
          var iterations = self.iterations
          var trials = 0
          SquareMatrixBenchmark_configure(&iterations, &trials)
          if isInitial {
            iterations = 1
            trials = 1
          } else {
            if currentConfig == .mps {
              // Too little sequential throughput.
              iterations = min(32, iterations)
            } else {
              trials *= trialsExtension
            }
          }
          
          let M = size
          let N = size
          let K = size
          var shapeA = A_trans ? [K, M] : [M, K]
          var shapeB = B_trans ? [N, K] : [K, N]
          var shapeC = [M, N]
          if let batchSize {
            shapeA = [batchSize] + shapeA
            if shapeA.last! % 3 == 0 {
              shapeB = [1] + shapeB
            }
            shapeC = [batchSize] + shapeC
          }
          
          let py_A = Tensor<Real>(
            shape: shapeA, randomUniform: 0..<1, backend: .numpy)
          let py_B = Tensor<Real>(
            shape: shapeB, randomUniform: 0..<1, backend: .numpy)
          
          let A = Tensor(copying: py_A)
          let B = Tensor(copying: py_B)
          var C = Tensor<Real>(zerosLike: shapeC)
          
          let backend = TensorBackend.default
          if isInitial {
            _ExecutionContext.executeExpression {
              backend.markFirstCommand()
              C.matmul(A, B, transposeA: A_trans, transposeB: B_trans)
              backend.markLastCommand()
              _ = backend.synchronize()
            }
          } else {
            var minTime: Double = .infinity
            for _ in 0..<trials {
              backend.markFirstCommand()
              for _ in 0..<iterations {
                C.matmul(A, B, transposeA: A_trans, transposeB: B_trans)
              }
              backend.markLastCommand()
              minTime = min(minTime, backend.synchronize())
            }
            
            if reportResults {
              var floatOps = 2 * M * N * K * iterations
              if let batchSize {
                floatOps *= batchSize
              }
              let flops = Double(floatOps) / minTime
              self.flops[currentConfig!]!.append(flops)
            }
          }
          
          if isInitial {
            let mps_A = Tensor(copying: py_A, backend: .mps)
            let mps_B = Tensor(copying: py_B, backend: .mps)
            var mps_C = Tensor<Real>(zerosLike: shapeC, backend: .mps)
            _ExecutionContext.withDefaultBackend(.mps) {
              _ExecutionContext.profileCommands {
                mps_C.matmul(
                  mps_A, mps_B, transposeA: A_trans, transposeB: B_trans)
              }
            }
            
            let params = EuclideanDistanceParameters(
              matrixK: K, batchSize: batchSize)
            if !C.isApproximatelyEqual(to: mps_C, parameters: params) {
              MPL_showComparison(
                actual: C, actualName: self.currentConfig!.name,
                expected: mps_C, expectedName: "MPS", parameters: params)
              fatalError("Tensors did not match.")
            }
          }
        }
        
        // Run the last matrix in the batch once to warm up, then actually start
        // benchmarking.
        innerLoop(size: sizes.upperBound - 1, reportResults: false)
        for size in sizes {
          innerLoop(size: size, reportResults: true)
        }
      }
      
      mutating func profile(
        granularity: Int, trialsExtension: Int, logProgress: Bool,
        A_trans: Bool, B_trans: Bool, batchSize: Int?
      ) {
        let reportGranularity = 16
        var start = self.sizes.lowerBound
        while start < self.sizes.upperBound {
          var end: Int
          if start + reportGranularity + 2 >= self.sizes.upperBound {
            end = self.sizes.upperBound
          } else {
            end = start + reportGranularity
          }
          
          let sectionSizes = start..<end
          for config in Config.fastConfigs {
            prepare(config: config)
            _profile(
              sizes: sectionSizes, granularity: granularity,
              trialsExtension: trialsExtension, isInitial: true,
              A_trans: A_trans, B_trans: B_trans, batchSize: batchSize)
            _profile(
              sizes: sectionSizes, granularity: granularity,
              trialsExtension: trialsExtension, isInitial: false,
              A_trans: A_trans, B_trans: B_trans, batchSize: batchSize)
            cleanup(config: config)
          }
          
          if logProgress {
            for size in sectionSizes {
              if size % granularity != 0 {
                continue
              }
              var message = "\(size)x\(size)x\(size)"
              if Real.self == Float.self {
                message += "xf32"
              } else {
                message += "xf16"
              }
              if let batchSize {
                message = "\(batchSize)x\(message)"
              }
              for config in Config.fastConfigs {
                let index = size - sizes.lowerBound
                let gflops = Int(flops[config]![index] / 1e9)
                message += " - \(config.name)"
                message += " \(gflops)"
              }
              print(message)
            }
          }
          
          if start + reportGranularity + 2 >= self.sizes.upperBound {
            break
          } else {
            start += reportGranularity
          }
        }
      }
    }
    
    var segments: [Segment] = [
      Segment(sizes: 1..<64, iterations: 256),
      Segment(sizes: 64..<128, iterations: 256),
      Segment(sizes: 128..<192, iterations: 256),
      Segment(sizes: 192..<256, iterations: 128),
      Segment(sizes: 256..<384, iterations: 64),
      Segment(sizes: 384..<512, iterations: 32),
      Segment(sizes: 512..<768, iterations: 16),
      Segment(sizes: 768..<1024, iterations: 8),
    ]
    if Real.self == Float.self || (batchSize ?? 1) > 1 {
      segments.append(Segment(sizes: 1024..<1537, iterations: 4))
    } else {
      segments.append(Segment(sizes: 1024..<1536, iterations: 4))
      segments.append(Segment(sizes: 1536..<2049, iterations: 2))
    }
    for i in 0..<segments.count {
      segments[i].profile(
        granularity: granularity, trialsExtension: trialsExtension,
        logProgress: logProgress,
        A_trans: A_trans, B_trans: B_trans, batchSize: batchSize)
    }
    
    func extract(config: Config) -> (size: [Int], gflops: [Double]) {
      var sizes: [Int] = []
      var speeds: [Double] = []
      for segment in segments {
        let flopsArray = segment.flops[config]!
        var sizeIndex = 0
        for size in segment.sizes {
          defer { sizeIndex += 1 }
          if size % granularity == 0 {
            sizes.append(size)
            speeds.append(flopsArray[sizeIndex])
          }
        }
      }
      return (sizes, speeds.map { $0 / 1e9 })
    }
    
    struct Extraction {
      var sizeArray: [Int]
      var gflopsArray: [Double]
      var title: String
      var style: String
      
      init(
        _ tuple: (size: [Int], gflops: [Double]),
        _ config: Config,
        style: String
      ) {
        self.sizeArray = tuple.size
        self.gflopsArray = tuple.gflops
        self.title = config.name
        self.style = style
      }
    }
    
    // green - MFA 48x48
    // blue - MFA 32x32
    // red - MPS
    let extractions: [Extraction] = [
      Extraction(extract(config: .mps), Config.mps, style: "-r"),
      Extraction(extract(config: .mfa32x32), Config.mfa32x32, style: "-b"),
      Extraction(extract(config: .mfa48x48), Config.mfa48x48, style: "-g"),
    ]
    let plt = PythonContext.global.plt
    for extraction in extractions {
      plt.plot(
        extraction.sizeArray, extraction.gflopsArray,
        extraction.style, label: extraction.title)
    }
    plt.legend(loc: "upper left")
    plt.xlim(0, extractions[0].sizeArray.last!)
    plt.ylim(0, MetalContext.global.infoDevice.flops / 1e9)
    plt.xlabel("Square Matrix Size")
    plt.ylabel("GFLOPS")
    
    var configRepr = (A_trans ? "T" : "N") + (B_trans ? "T" : "N")
    if let batchSize {
      configRepr += ", Batched, \(batchSize)xA"
    }
    #if DEBUG
    let debugWarning = "(NOT USABLE FOR CI)"
    #else
    let debugWarning = ""
    #endif
    if Real.self == Float.self {
      plt.title("Float32 Utilization (\(configRepr)) \(debugWarning)")
    } else {
      plt.title("Float16 Utilization (\(configRepr)) \(debugWarning)")
    }
    plt.show()
  }
}
