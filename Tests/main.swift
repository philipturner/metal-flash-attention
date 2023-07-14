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
