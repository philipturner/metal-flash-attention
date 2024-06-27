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
  print("Hello, console.")
  
  // Next, implement "naive attention" with the unified GEMM kernel. Measure
  // performance of the forward and backward pass with various problem configs.
  //
  // This will require a few custom GPU kernels:
  // - softmax
  //   - load all of the matrix elements into registers
  //   - requires knowledge of problem size up-front
  // - D[i] term
  // - dS elementwise
  //   - find a workable way to copy the entire unified GEMM kernel source,
  //     without adding technical debt to the minimal implementation in-tree
  //   - initialize accumulator for dP as D[i]
  //   - "use async store" becomes "load previous C value with async copy"
  //   - C will always be written to threadgroup memory
  
  // Design a kernel for the D term.
  // - Make a test for correctness.
  // - Ensure the latency is negligible for all reasonable problem sizes.
}
#endif

// A configuration for a D[i] term kernel.
struct DTermDescriptor {
  // Limited to 16 bits, because the preceding softmax cannot support anything
  // larger.
  var matrixDimensions: (N: UInt16, D: UInt16)?
  
  // Supports any combination of mixed precisions, with the following
  // constraints:
  // - dO cannot be FP16
  // - O cannot be BF16
  var memoryPrecisions: (dO: GEMMOperandPrecision, O: GEMMOperandPrecision)?
}

// Generates the D[i] terms.
struct DTermKernel {
  var source: String = ""
  
  var threadgroupSize: UInt16
  
  init(descriptor: DTermDescriptor) {
    guard let matrixDimensions = descriptor.matrixDimensions,
          let memoryPrecisions = descriptor.memoryPrecisions else {
      fatalError("Descriptor was incomplete.")
    }
    self.threadgroupSize = 32
    
    // Find the MSL keywords for each precision.
    guard memoryPrecisions.dO != .FP16,
          memoryPrecisions.O != .BF16 else {
      fatalError("Invalid precision for output matrix.")
    }
    let memoryNameDerivativeO = memoryPrecisions.dO.name
    let memoryNameO = memoryPrecisions.O.name
    
    // Allocate enough registers to cache the entire matrix row.
    var paddedD = matrixDimensions.D + 32 - 1
    paddedD = (paddedD / 32) * 32
    
    source = """
#include <metal_stdlib>
using namespace metal;

kernel void generate(
  device \(memoryNameDerivativeO) *dO [[buffer(0)]],
  device \(memoryNameO) *O [[buffer(1)]],

  uint gid [[threadgroup_position_in_grid]],
  ushort lane_id [[thread_index_in_simdgroup]])
{
  \(memoryNameDerivativeO) dO_elements[\(paddedD / 32)];
  \(memoryNameO) O_elements[\(paddedD / 32)];
  
  // TODO: Implement the optimization from softmax, which removes the overhead
  // of indexing in the loop.
  auto dO_src = dO + gid * \(matrixDimensions.D);
  auto O_src = O + gid * \(matrixDimensions.D);
  for (uint d = lane_id; d < \(matrixDimensions.D); c += 32) {
    
  }
}

"""
  }
}
