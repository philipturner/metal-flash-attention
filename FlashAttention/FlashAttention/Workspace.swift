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
  //
  // Task 3:
  // - Make a copy of the in-tree GEMM kernel, which fuses some operations
  //   during computation of dS.
  // - Alternatively, modify 'GEMMKernel' to enable fused operations on the
  //   accumulator. This would require heavy testing to ensure no regressions.
  
  // 2) Reproduce the Laplacian test, but remove the profiling parts. Make it
  //    just a correctness test. Set the current value of P to a constant
  //    multiplicative factor.
  // 3) Include the D[i] term when initializing the accumulator.
  
  // Set up a correctness test with matrix dimensions typical for attention.
  let N: Int = 32
  let D: Int = 8
  
  // Create the GEMM kernel.
  var gemmDesc = GEMMDescriptor()
  gemmDesc.matrixDimensions = (UInt32(N), UInt32(N), UInt32(D))
  gemmDesc.memoryPrecisions = (.BF16, .FP16, .BF16)
  gemmDesc.transposeState = (false, true)
  var kernelDesc = GEMMKernelDescriptor(descriptor: gemmDesc)
  kernelDesc.device = nil
  kernelDesc.preferAsyncStore = nil
  let kernel = DerivativeSoftmaxKernel(descriptor: kernelDesc)
  print(kernel.source)
  
  // Create the reference implementation.
  var networkDesc = NetworkDescriptor()
  networkDesc.N = N
  networkDesc.D = D
  let network = Network(descriptor: networkDesc)
  
  // Generate the attention matrices with the reference implementation.
  var matrixP: [Float] = []
  var derivativeP: [Float] = []
  var derivativeS: [Float] = []
  for rowID in 0..<N {
    let matrixPRow = network.createMatrixPRow(rowID: rowID)
    let derivativePRow = network.createDerivativePRow(rowID: rowID)
    let derivativeSRow = network.createDerivativeSRow(rowID: rowID)
    matrixP += matrixPRow
    derivativeP += derivativePRow
    derivativeS += derivativeSRow
  }
  
  // Displays the first 8x8 block of the matrix.
  func printMatrix(_ matrix: [Float]) {
    for r in 0..<8 {
      for c in 0..<8 {
        let matrixAddress = r * N + c
        let matrixValue = matrix[matrixAddress]
        var repr = String(format: "%.3f", matrixValue)
        while repr.count < 8 {
          repr = " " + repr
        }
        print(repr, terminator: " ")
      }
      print()
    }
  }
  
  print()
  print("P:")
  printMatrix(matrixP)
  
  print()
  print("dP:")
  printMatrix(derivativeP)
  
  print()
  print("dS:")
  printMatrix(derivativeS)
}
#endif
