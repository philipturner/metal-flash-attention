//
//  PerformanceTests.swift
//  MetalFlashAttention
//
//  Created by Philip Turner on 6/27/23.
//

import Metal

class PerformanceTests: MFATestCase {
  override class func typeDescription() -> String {
    "PerformanceTests"
  }
  
  override func runVeryLongTests() {
    // Tests the precision you set as the global testing precision. For a quick
    // smoke test, you can set a larger granularity.
    testGEMMSpeed(granularity: 1, logProgress: true)
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
          MFA_GEMM.functionConstants["K_simd"] = UInt16(16)
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
      mutating func _profile(granularity: Int, isInitial: Bool) {
        func innerLoop(size: Int, reportResults: Bool) {
          var iterations = self.iterations
          var trials = 0
          SquareMatrixBenchmark_configure(&iterations, &trials)
          if isInitial {
            iterations = 1
            trials = 1
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
          }
          
          // If `isInitial`, validate that the result matches a tensor generated
          // on NumPy.
          if isInitial {
            var py_C = Tensor<Real>(zerosLike: [M, N], backend: .numpy)
            TensorBackend.numpy.withGhostExecution {
              py_C.matmul(py_A, py_B)
            }
            py_C.matmul(py_A, py_B)
            
            let params = EuclideanDistanceParameters(matrixK: K)
            if !C.isApproximatelyEqual(to: py_C, parameters: params) {
              MPL_showComparison(
                actual: C, actualName: self.currentConfig!.name,
                expected: py_C, expectedName: "NumPy", parameters: params)
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
        for config in Config.allCases {
          prepare(config: config)
          _profile(granularity: granularity, isInitial: true)
          _profile(granularity: granularity, isInitial: false)
          cleanup(config: config)
        }
        
        if logProgress {
          print("Profiling from \(sizes.lowerBound) to \(sizes.upperBound)")
          
          // TODO: While logging, simply print the backend and GFLOPS.
          // TODO: Then, erase the statement above.
        }
      }
    }
    
    // TODO: Generate a matplotlib line chart from this data.
    var segments: [Segment] = [
      Segment(sizes: 1..<64, iterations: 1024),
      Segment(sizes: 64..<128, iterations: 512),
      Segment(sizes: 128..<192, iterations: 256),
      Segment(sizes: 192..<256, iterations: 128),
      Segment(sizes: 256..<384, iterations: 64),
      Segment(sizes: 384..<512, iterations: 32),
      Segment(sizes: 512..<768, iterations: 16),
      Segment(sizes: 768..<1024, iterations: 8),
    ]
    if Real.self == Float.self {
      segments.append(Segment(sizes: 1024..<1537, iterations: 4))
    } else {
      segments.append(Segment(sizes: 1024..<1536, iterations: 4))
      segments.append(Segment(sizes: 1536..<2049, iterations: 2))
    }
    for i in 0..<segments.count {
      segments[i].profile(granularity: granularity, logProgress: logProgress)
    }
  }
}
