//
//  main.swift
//  MetalFlashAttention
//
//  Created by Philip Turner on 6/23/23.
//

import Foundation

// Ensure all contexts load correctly.
_ = MetalContext.global
_ = PythonContext.global

// Show an example comparing results from MPSGraph and NumPy.
//testImports()

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
// M_splits = 2
// N_splits = 2
// K_splits = 1
//
// Configs:
// - M_simd = 32
// - N_simd = 32
// - K_simd = 32
//
// - M_simd = 48
// - N_simd = 48
// - K_simd = 32 (HGEMM), 24 (SGEMM)
// TODO: Function that runs tests.

// TODO: Run a rigorous test suite, covering all the functionality you added to
// MFA. Randomly generate a vastly number of hyperparameter combinations and
// search for correctness regressions.

// TODO: Search for performance regressions, ensure everything is faster or 90%
// as fast as MPSGraph, generate graphs from the Nvidia Stream-K paper.
