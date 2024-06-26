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
  // Task 2:
  // - Design a theoretically optimal softmax kernel.
  //
  // Task 3:
  // - Make a copy of the in-tree GEMM kernel, which fuses some operations
  //   during computation of dS.
  
  // Define the problem dimensions.
  let N: Int = 10
  let D: Int = 3
  
  var networkDesc = NetworkDescriptor()
  networkDesc.N = N
  networkDesc.D = D
  let network = Network(descriptor: networkDesc)
  
  let matrixSRow = network.createMatrixSRow(rowID: 0)
  let matrixPRow = network.createMatrixPRow(rowID: 0)
  
  func printRow(_ row: [Float]) {
    for element in row {
      var repr = String(format: "%.3f", element)
      while repr.count < 8 {
        repr = " " + repr
      }
      print(repr, terminator: " ")
    }
    print()
  }
  
  print()
  print("S[0]")
  printRow(matrixSRow)
  print(matrixSRow.reduce(0, +))
  
  print()
  print("P[0]")
  printRow(matrixPRow)
  print(matrixPRow.reduce(0, +))
}
