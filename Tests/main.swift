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

// Eventually encapsulate this in the very long "GEMM speed" test. The test will
// pause until you close the window showing performance.
//
// TODO: Run the performance tests in release mode and without API validation,
// which might improve sequential throughput of MPSGraph.
MFATestCase.runTests(speed: .veryLong)

// Currently, the tests only cover:
// M = 1 - 1536/2048
// N = 1 - 1536/2048
// K = 1 - 1536/2048
// A_trans = false
// B_trans = false
// alpha = 1
// beta = 0
// batched = false
// fused_activation = false
// K_splits = 1
//
// TODO: Run a rigorous test suite, covering all the functionality you added to
// MFA. Randomly generate a vast number of hyperparameter combinations and look
// for correctness regressions.
//
// TODO: Use Matplotlib to visualize performance, search for performance
// regressions, ensure everything is faster or 90% as fast as MPSGraph, generate
// graphs from the Nvidia Stream-K paper. This will happen in the distant
// future; correctness is the current priority.
