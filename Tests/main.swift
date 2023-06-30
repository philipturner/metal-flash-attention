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
// TODO: Generate graphs from the Nvidia Stream-K paper. This will happen in the
// distant future; correctness is the current priority.
