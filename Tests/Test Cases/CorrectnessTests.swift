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
    
    // 25 - batch 1, NN
    // 50 - batch 1, NN/NT/TN/TT
    // 75 - batch 2-8 for either operand
    let numTrials = 75 // 150
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
    
    let allRandomTransposes: [(Bool, Bool)] = (0..<numTrials).map { i in
      if i < 25 {
        return (false, false)
      } else if i < 75 {
        // Only B_trans supported right now.
        return (Bool.random(), Bool.random())
      } else {
        fatalError("Unsupported")
      }
    }
    
    func testRandomSize(index: Int, ghost: Bool) {
      let randomInts = allRandomInts[index]
      let randomTransposes = allRandomTransposes[index]
      
      let M = randomInts[0]
      let N = randomInts[1]
      let K = randomInts[2]
      let A_trans = randomTransposes.0
      let B_trans = randomTransposes.1
      let DTypeRepr = (Real.self == Float.self) ? "f32" : "f16"
      let transRepr = (A_trans ? "T" : "N") + (B_trans ? "T" : "N")
      
      let mps_A = Tensor<Real>(
        shape: A_trans ? [K, M] : [M, K],
        randomUniform: 0..<1, backend: .mps)
      let mps_B = Tensor<Real>(
        shape: B_trans ? [N, K] : [K, N],
        randomUniform: 0..<1, backend: .mps)
      var mps_C = Tensor<Real>(zerosLike: [M, N], backend: .mps)
      
      let mfa_A = Tensor(copying: mps_A, backend: .mfa)
      let mfa_B = Tensor(copying: mps_B, backend: .mfa)
      var mfa_C = Tensor(copying: mps_C, backend: .mfa)
      
      func act(A: Tensor<Real>, B: Tensor<Real>, C: inout Tensor<Real>) {
        C.matmul(A, B, transposeA: A_trans, transposeB: B_trans)
      }
      
      if ghost {
        _ExecutionContext.withDefaultBackend(.mps) {
          TensorBackend.default.withGhostExecution {
            TensorBackend.default.markFirstCommand()
            act(A: mps_A, B: mps_B, C: &mps_C)
            TensorBackend.default.markLastCommand()
            _ = TensorBackend.default.synchronize()
          }
        }
        _ExecutionContext.withDefaultBackend(.mfa) {
          TensorBackend.default.withGhostExecution {
            TensorBackend.default.markFirstCommand()
            act(A: mfa_A, B: mfa_B, C: &mfa_C)
            TensorBackend.default.markLastCommand()
            _ = TensorBackend.default.synchronize()
          }
        }
      } else {
        _ExecutionContext.withDefaultBackend(.mps) {
          TensorBackend.default.markFirstCommand()
          act(A: mps_A, B: mps_B, C: &mps_C)
          TensorBackend.default.markLastCommand()
          _ = TensorBackend.default.synchronize()
        }
        _ExecutionContext.withDefaultBackend(.mfa) {
          TensorBackend.default.markFirstCommand()
          act(A: mfa_A, B: mfa_B, C: &mfa_C)
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
          print("Passed test: \(M)x\(N)x\(K)x\(DTypeRepr) (\(transRepr))")
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
  
//  let M = 100
//  let N = 50
//  let K = 25
//
//  typealias Real = Float32
//
//  let py_A = Tensor<Real>(shape: [K, M], randomUniform: 0..<1, backend: .numpy)
//  let py_B = Tensor<Real>(shape: [K, N], randomUniform: 0..<1, backend: .numpy)
//  var py_C = Tensor<Real>(zerosLike: [M, N], backend: .numpy)
//  _ExecutionContext.withDefaultBackend(.numpy) {
//    _ExecutionContext.profileCommands {
//      py_C.matmul(py_A, py_B, transposeA: true)
//    }
//  }
//
//  let mps_A = Tensor(copying: py_A, backend: .mps)
//  let mps_B = Tensor(copying: py_B, backend: .mps)
//  var mps_C = Tensor<Real>(zerosLike: [M, N], backend: .mps)
//  _ExecutionContext.withDefaultBackend(.mps) {
//    _ExecutionContext.profileCommands {
//      mps_C.matmul(mps_A, mps_B, transposeA: true)
//    }
//  }
//
//  let mfa_A = Tensor(copying: py_A, backend: .mfa)
//  let mfa_B = Tensor(copying: py_B, backend: .mfa)
//  var mfa_C = Tensor<Real>(zerosLike: [M, N], backend: .mfa)
//  _ExecutionContext.withDefaultBackend(.mfa) {
//    _ExecutionContext.profileCommands {
//      mfa_C.matmul(mfa_A, mfa_B, transposeA: true)
//    }
//  }
//
//  MPL_showBackends(
//    mfa: mfa_C, mps: mps_C, numpy: py_C,
//    parameters: .init(matrixK: K))
}
