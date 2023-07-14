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

// TODO: Before testing attention, ensure triangular and block-sparse masks look
// correct, for both FP16 and FP32.

MFATestCase.runTests(speed: .quick)
