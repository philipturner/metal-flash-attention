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
  // Task 2:
  // - Design a theoretically optimal softmax kernel.
  //
  // Task 3:
  // - Make a copy of the in-tree GEMM kernel, which fuses some operations
  //   during computation of dS.
  // - Alternatively, modify 'GEMMKernel' to enable fused operations on the
  //   accumulator. This would require heavy testing to ensure no regressions.
  
  // Define the problem dimensions.
  let N: Int = 10
  let D: Int = 3
  
  var networkDesc = NetworkDescriptor()
  networkDesc.N = N
  networkDesc.D = D
  let network = Network(descriptor: networkDesc)
  
  var matrixS: [Float] = []
  var matrixP: [Float] = []
  for rowID in 0..<N {
    let matrixSRow = network.createMatrixSRow(rowID: rowID)
    let matrixPRow = network.createMatrixPRow(rowID: rowID)
    matrixS += matrixSRow
    matrixP += matrixPRow
  }
  
  // Displays a matrix with dimensions N * N.
  func printMatrix(_ matrix: [Float]) {
    for r in 0..<N {
      for c in 0..<N {
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
  print("S")
  printMatrix(matrixS)
  
  print()
  print("P")
  printMatrix(matrixP)
  
  // Create the kernel.
  var softmaxDesc = SoftmaxDescriptor()
  softmaxDesc.threadgroupSize = 128
  softmaxDesc.memoryPrecision = .FP32
  softmaxDesc.matrixDimensions = (UInt16(N), UInt16(D))
  let softmaxKernel = SoftmaxKernel(descriptor: softmaxDesc)
  
  // Create the pipeline state object.
  var pipeline: MTLComputePipelineState
  do {
    let library = try! MTLContext.global.device
      .makeLibrary(source: softmaxKernel.source, options: nil)
    let computeFunction = library.makeFunction(name: "softmax")!
    pipeline = try! MTLContext.global.device
      .makeComputePipelineState(function: computeFunction)
  }
  
  // Create the buffer.
  let attentionMatrixBuffer = MTLContext.global
    .createBuffer(matrixS, softmaxDesc.memoryPrecision!)
  
  do {
    // Encode the GPU command.
    let commandBuffer = MTLContext.global.commandQueue.makeCommandBuffer()!
    let encoder = commandBuffer.makeComputeCommandEncoder()!
    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(attentionMatrixBuffer, offset: 0, index: 0)
    do {
      let gridSize = MTLSize(
        width: Int(N), height: 1, depth: 1)
      let groupSize = MTLSize(
        width: Int(softmaxKernel.threadgroupSize), height: 1, depth: 1)
      encoder.dispatchThreadgroups(
        gridSize, threadsPerThreadgroup: groupSize)
    }
    encoder.endEncoding()
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
  }
  
  // Copy the results.
  var result = [Float](repeating: .zero, count: N * N)
  do {
    let precision = softmaxDesc.memoryPrecision!
    let raw = attentionMatrixBuffer.contents()
    for r in 0..<N {
      for c in 0..<N {
        let address = r * N + c
        var entry32: Float
        
        switch precision {
        case .FP32:
          let casted = raw.assumingMemoryBound(to: Float.self)
          entry32 = casted[address]
        case .FP16:
          let casted = raw.assumingMemoryBound(to: Float16.self)
          let entry16 = casted[address]
          entry32 = Float(entry16)
        case .BF16:
          let casted = raw.assumingMemoryBound(to: UInt16.self)
          let entry16 = casted[address]
          let entry16x2 = SIMD2<UInt16>(.zero, entry16)
          entry32 = unsafeBitCast(entry16x2, to: Float.self)
        }
        result[address] = entry32
      }
    }
  }
  
  print()
  print("result")
  printMatrix(result)
}
#endif
