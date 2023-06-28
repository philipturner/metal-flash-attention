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


let ctx = PythonContext.global
let x = ctx.np.array([[0, 1], [2, 3]], dtype: ctx.np.int32)
print(x)
print(x.ctypes.data)

let pyInt = x.ctypes.data
let integer = UInt(Int(pyInt)!)
print(integer)

let pointer = UnsafeMutablePointer<Int32>(bitPattern: integer)!
print(pointer)
print(pointer[0])
print(pointer[1])
print(pointer[2])
print(pointer[3])

exit(0)

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

// TODO: Use Matplotlib to visualize performance, search for performance
// regressions, ensure everything is faster or 90% as fast as MPSGraph, generate
// graphs from the Nvidia Stream-K paper.
//
// This will happen in the distant future; correctness is the current priority.
// TODO: However, we will at least ensure there are no register spills (GEMM
// must execute reasonably fast: >7000 GFLOPS for all HGEMM sizes >1000).
MFATestCase.runTests(speed: .veryLong)
