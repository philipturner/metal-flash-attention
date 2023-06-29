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
    testGEMMSpeed(granularity: 4, logProgress: true)
  }
  
  // Covers the entire range of square matrix sizes, as well as differences
  // between MFA 32x32, MFA 48x48, and MPS.
  func testGEMMSpeed(granularity: Int, logProgress: Bool) {
    precondition(granularity.nonzeroBitCount == 1)
    
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
        case .mfa48x48: return "MFA_48x48"
        case .mfa32x32: return "MFA_32x32"
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
      mutating func _profile(sizes: Range<Int>, granularity: Int, isInitial: Bool) {
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
            }
          }
          
          let M = size
          let N = size
          let K = size
          let py_A = Tensor<Real>(
            shape: [M, K], randomUniform: 0..<1, backend: .numpy)
          let py_B = Tensor<Real>(
            shape: [K, N], randomUniform: 0..<1, backend: .numpy)
          
          let A = Tensor(copying: py_A)
          let B = Tensor(copying: py_B)
          var C = Tensor<Real>(zerosLike: [M, N])
          
          let backend = TensorBackend.default
          if isInitial {
            _ExecutionContext.executeExpression {
              backend.markFirstCommand()
              C.matmul(A, B)
              backend.markLastCommand()
              _ = backend.synchronize()
            }
          } else {
            var minTime: Double = .infinity
            for _ in 0..<trials {
              backend.markFirstCommand()
              for _ in 0..<iterations {
                C.matmul(A, B)
              }
              backend.markLastCommand()
              minTime = min(minTime, backend.synchronize())
            }
            
            if reportResults {
              let floatOps = 2 * M * N * K * iterations
              let flops = Double(floatOps) / minTime
              self.flops[currentConfig!]!.append(flops)
            }
          }
          
          if isInitial {
            let mps_A = Tensor(copying: py_A, backend: .mps)
            let mps_B = Tensor(copying: py_B, backend: .mps)
            var mps_C = Tensor<Real>(zerosLike: [M, N], backend: .mps)
            _ExecutionContext.withDefaultBackend(.mps) {
              _ExecutionContext.profileCommands {
                mps_C.matmul(mps_A, mps_B)
              }
            }
            
            let params = EuclideanDistanceParameters(matrixK: K)
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
      
      mutating func profile(granularity: Int, logProgress: Bool) {
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
            _profile(sizes: sectionSizes, granularity: granularity, isInitial: true)
            _profile(sizes: sectionSizes, granularity: granularity, isInitial: false)
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
    
    // TODO: Generate a matplotlib line chart from this data.
    var segments: [Segment] = [
      Segment(sizes: 1..<64, iterations: 128),
      Segment(sizes: 64..<128, iterations: 128),
      Segment(sizes: 128..<192, iterations: 128),
      Segment(sizes: 192..<256, iterations: 64),
      Segment(sizes: 256..<384, iterations: 32),
      Segment(sizes: 384..<512, iterations: 16),
      Segment(sizes: 512..<768, iterations: 8),
      Segment(sizes: 768..<1024, iterations: 4),
    ]
//    if Real.self == Float.self {
//      segments.append(Segment(sizes: 1024..<1537, iterations: 2))
//    } else {
//      segments.append(Segment(sizes: 1024..<1536, iterations: 2))
//      segments.append(Segment(sizes: 1536..<2049, iterations: 1))
//    }
    for i in 0..<segments.count {
      segments[i].profile(granularity: granularity, logProgress: logProgress)
    }
  }
}
