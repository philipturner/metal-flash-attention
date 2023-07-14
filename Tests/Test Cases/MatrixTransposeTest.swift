//
//  MatrixTransposeTest.swift
//  MetalFlashAttention
//
//  Created by Philip Turner on 7/13/23.
//

import Foundation
import PythonKit

func showMatrixTransposeTest() {
  let M = 100
  let N = 50
  let K = 25
  let batchSize: Int? = 2
  
  var shapeA: [Int]
  var shapeB: [Int]
  var shapeC: [Int]
  if let batchSize {
    shapeA = [3, batchSize, K, M]
    shapeB = [3, batchSize, K, N]
    shapeC = [3, batchSize, M, N]
  } else {
    shapeA = [K, M]
    shapeB = [K, N]
    shapeC = [M, N]
  }
  
  typealias Real = Float32
  
  let py_A = Tensor<Real>(shape: shapeA, randomUniform: 0..<1, backend: .numpy)
  let py_B = Tensor<Real>(shape: shapeB, randomUniform: 0..<1, backend: .numpy)
  var py_C = Tensor<Real>(zerosLike: shapeC, backend: .numpy)
  _ExecutionContext.withDefaultBackend(.numpy) {
    _ExecutionContext.profileCommands {
      py_C.matmul(py_A, py_B, transposeA: true)
    }
  }
  
  let mps_A = Tensor(copying: py_A, backend: .mps)
  let mps_B = Tensor(copying: py_B, backend: .mps)
  var mps_C = Tensor<Real>(zerosLike: shapeC, backend: .mps)
  _ExecutionContext.withDefaultBackend(.mps) {
    _ExecutionContext.profileCommands {
      mps_C.matmul(mps_A, mps_B, transposeA: true)
    }
  }
  
  let mfa_A = Tensor(copying: py_A, backend: .mfa)
  let mfa_B = Tensor(copying: py_B, backend: .mfa)
  var mfa_C = Tensor<Real>(zerosLike: shapeC, backend: .mfa)
  _ExecutionContext.withDefaultBackend(.mfa) {
    _ExecutionContext.profileCommands {
      mfa_C.matmul(mfa_A, mfa_B, transposeA: true)
    }
  }
  
  MPL_showBackends(
    mfa: mfa_C, mps: mps_C, numpy: py_C,
    parameters: .init(matrixK: K, batchSize: batchSize),
    slice: PythonObject(tupleOf: 0, 0))
  MPL_showBackends(
    mfa: mfa_C, mps: mps_C, numpy: py_C,
    parameters: .init(matrixK: K, batchSize: batchSize),
    slice: PythonObject(tupleOf: 0, 1))
  MPL_showBackends(
    mfa: mfa_C, mps: mps_C, numpy: py_C,
    parameters: .init(matrixK: K, batchSize: batchSize),
    slice: PythonObject(tupleOf: 1, 0))
}
