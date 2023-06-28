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

let py_matrixA = Tensor<Real>(
  shape: [20, 20], randomUniform: 0..<1, backend: .numpy)
let py_matrixB = Tensor<Real>(
  shape: [20, 20], randomUniform: 0..<1, backend: .numpy)
let py_matrixC = Tensor<Real>(
  shape: [20, 20], randomUniform: 0..<1, backend: .numpy)

#if false
MPL_showBackends(
  mfa: py_matrixA,
  mps: py_matrixB,
  numpy: py_matrixC,
  parameters: .init(matrixK: 2))
#else
MPL_showComparison(
  actual: py_matrixA,
  expected: py_matrixB,
  parameters: .init(matrixK: 2))
#endif

#if false
let (matrices, parameters) = MPL_exampleMatrices(withGEMM: true)

MPL_showBackends(
  mfa: matrices[0],
  mps: matrices[1],
  numpy: matrices[2],
  parameters: parameters)

MPL_showComparison(
  actual: matrices[0],
  expected: matrices[1],
  parameters: parameters)
#endif

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
#if false
MFATestCase.runTests(speed: .veryLong)
#endif
