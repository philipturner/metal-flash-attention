//
//  Workspace.swift
//  FlashAttention
//
//  Created by Philip Turner on 6/20/24.
//

import Metal
import QuartzCore

/// The repo author's own workspace for running tests and developing kernels.
/// The contents of this function have no meaning, and ideally will be blank
/// when the 'main' branch is in a stable state. Clients can utilize this
/// function to script tests in their fork.

// Tasks:
// - Fire up the new square attention test, using the default parameters. [DONE]
// - Run a benchmark over the head dimension. [DONE]
// - Run the same benchmark, but with the parameters. [DONE]
// - Tweak parameters that give the wrong performance on M1. [DONE]
// - Tweak parameters that give the wrong performance on M3.
//   - Expand the benchmarked range to D=384.
// - Test the revised parameters at a different sequence length.
