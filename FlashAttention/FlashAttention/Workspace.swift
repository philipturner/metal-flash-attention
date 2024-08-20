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
// - Check for a performance improvement from the parameters.
// - Check for a performance improvement in the remaining ones.
// - Tweak parameters that give the wrong performance.
// - Test the revised parameters at a different sequence length.
