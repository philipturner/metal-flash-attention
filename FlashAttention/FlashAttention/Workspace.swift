//
//  Workspace.swift
//  FlashAttention
//
//  Created by Philip Turner on 6/20/24.
//

import Metal
import QuartzCore

#if true
/// The repo author's own workspace for running tests and developing kernels.
/// The contents of this function have no meaning, and ideally will be blank
/// when the 'main' branch is in a stable state. Clients can utilize this
/// function to script tests in their fork.
func executeScript() {
  // Deferring profiling against naive attention to a later date (it must be
  // done for accurate insights into performance). For now, working on the
  // greatest bottleneck: getting any test at all, of attention being done
  // entirely on the GPU.
  print("Hello, console.")
}


#endif
