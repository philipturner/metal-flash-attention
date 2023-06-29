//
//  main.swift
//  MetalFlashAttention
//
//  Created by Philip Turner on 6/23/23.
//

import AppleGPUInfo
import Foundation
import PythonKit

// Ensure all contexts load correctly.
_ = MetalContext.global
_ = PythonContext.global

// Global setting for the precision used in tests.
typealias Real = Float16

let M = 47
let N = 47
let K = 51
let parameters = EuclideanDistanceParameters(matrixK: K)

let py_A = Tensor<Real>(
  shape: [M, K], randomUniform: 0..<1, backend: .numpy)
let py_B = Tensor<Real>(
  shape: [K, N], randomUniform: 0..<1, backend: .numpy)

func _makeTensor(on backend: TensorBackend) -> Tensor<Real> {
  backend.markFirstCommand()
  let A = Tensor<Real>(copying: py_A)
  let B = Tensor<Real>(copying: py_B)
  var C = Tensor<Real>(zerosLike: [M, N])
  C.matmul(A, B)
  backend.markLastCommand()
  _ = backend.synchronize()
  return C
}

func makeTensor(on backend: TensorBackend) -> Tensor<Real> {
  _ExecutionContext.defaultBackend = backend
  let output = _ExecutionContext.executeExpression {
    _makeTensor(on: backend)
  }
  _ExecutionContext.defaultBackend = .numpy
  return output
}

let numpy = makeTensor(on: .numpy)
var mps = makeTensor(on: .mps)
let mfa = makeTensor(on: .mfa)

MPL_showBackends(mfa: mfa, mps: mps, numpy: numpy, parameters: parameters)
//MPL_showComparison(actual: mfa, expected: mps, parameters: parameters)
precondition(mfa.isApproximatelyEqual(to: mps, parameters: parameters))

// Currently, the tests only cover:
// M = 1 - 1000
// N = 1 - 1000
// K = 1 - 1000
// A_trans = false
// B_trans = false
// alpha = 1
// beta = 0
// batched = false
// fused_activation = false
//
// TODO: Run a rigorous test suite, covering all the functionality you added to
// MFA. Randomly generate a vast number of hyperparameter combinations and look
// for correctness regressions.
//
// TODO: Use Matplotlib to visualize performance, search for performance
// regressions, ensure everything is faster or 90% as fast as MPSGraph, generate
// graphs from the Nvidia Stream-K paper.
//
// This will happen in the distant future; correctness is the current priority.
// TODO: However, we will at least ensure there are no register spills (GEMM
// must execute reasonably fast: >7000 GFLOPS for all HGEMM sizes >1000).
MFATestCase.runTests(speed: .veryLong)
