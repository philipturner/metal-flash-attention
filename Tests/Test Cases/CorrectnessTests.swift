//
//  CorrectnessTests.swift
//  MetalFlashAttention
//
//  Created by Philip Turner on 6/29/23.
//

import Metal
import QuartzCore

class CorrectnessTests: MFATestCase {
  override class func typeDescription() -> String {
    "CorrectnessTests"
  }
  
  override func runQuickTests() {
    testRandomMatrices(logProgress: true)
  }
  
  func testRandomMatrices(logProgress: Bool) {
    let start = CACurrentMediaTime()
    
    let numTrials = 100
    let maxMatrixDimension = 1000
    
    // Create a biased random distribution that favors smaller numbers. Take the
    // uniform distribution, then cube the results.
    let allRandomInts: [SIMD3<Int>] = (0..<numTrials).map { _ in
      var randomVecFloat = SIMD3<Float>.random(in: 0..<1)
      randomVecFloat = randomVecFloat * randomVecFloat * randomVecFloat
      var randomInts = SIMD3<Int>(randomVecFloat * Float(maxMatrixDimension))
      randomInts.replace(with: .one, where: randomInts .== .zero)
      return randomInts
    }
    
    func testRandomSize(index: Int, ghost: Bool) {
      let randomInts = allRandomInts[index]
      
      let M = randomInts[0]
      let N = randomInts[1]
      let K = randomInts[2]
      let DTypeRepr = (Real.self == Float.self) ? "f32" : "f16"
      
      let mps_A = Tensor<Real>(
        shape: [M, K], randomUniform: 0..<1, backend: .mps)
      let mps_B = Tensor<Real>(
        shape: [K, N], randomUniform: 0..<1, backend: .mps)
      var mps_C = Tensor<Real>(zerosLike: [M, N], backend: .mps)
      
      let mfa_A = Tensor(copying: mps_A, backend: .mfa)
      let mfa_B = Tensor(copying: mps_B, backend: .mfa)
      var mfa_C = Tensor(copying: mps_C, backend: .mfa)
      
      if ghost {
        _ExecutionContext.withDefaultBackend(.mps) {
          TensorBackend.default.withGhostExecution {
            TensorBackend.default.markFirstCommand()
            mps_C.matmul(mps_A, mps_B)
            TensorBackend.default.markLastCommand()
            _ = TensorBackend.default.synchronize()
          }
        }
        _ExecutionContext.withDefaultBackend(.mfa) {
          TensorBackend.default.withGhostExecution {
            TensorBackend.default.markFirstCommand()
            mfa_C.matmul(mfa_A, mfa_B)
            TensorBackend.default.markLastCommand()
            _ = TensorBackend.default.synchronize()
          }
        }
      } else {
        _ExecutionContext.withDefaultBackend(.mps) {
          TensorBackend.default.markFirstCommand()
            mps_C.matmul(mps_A, mps_B)
          TensorBackend.default.markLastCommand()
          _ = TensorBackend.default.synchronize()
        }
        _ExecutionContext.withDefaultBackend(.mfa) {
          TensorBackend.default.markFirstCommand()
          mfa_C.matmul(mfa_A, mfa_B)
          TensorBackend.default.markLastCommand()
          _ = TensorBackend.default.synchronize()
        }
        
        let params = EuclideanDistanceParameters(matrixK: K)
        if !mfa_C.isApproximatelyEqual(to: mps_C, parameters: params) {
          MPL_showComparison(
            actual: mfa_C, actualName: "MFA",
            expected: mps_C, expectedName: "MPS", parameters: params)
          fatalError("Tensors did not match.")
        }
        if logProgress {
          print("Passed test: \(M)x\(N)x\(K)x\(DTypeRepr)")
        }
      }
    }
    
    for i in 0..<numTrials {
      testRandomSize(index: i, ghost: true)
    }
    for i in 0..<numTrials {
      testRandomSize(index: i, ghost: false)
    }
    
    let end = CACurrentMediaTime()
    let repr = String(format: "%.3f", end - start)
    print("Finished 'testRandomMatrices' in \(repr) seconds.")
  }
}
