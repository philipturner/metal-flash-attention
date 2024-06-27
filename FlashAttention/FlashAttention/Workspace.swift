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
  
  // Set up a correctness test with matrix dimensions typical for attention.
  let N: Int = 320
  let D: Int = 800
  
  // Create the GEMM kernel.
  var gemmDesc = GEMMDescriptor()
  gemmDesc.matrixDimensions = (UInt32(N), UInt32(N), UInt32(D))
  gemmDesc.memoryPrecisions = (.BF16, .FP16, .BF16)
  gemmDesc.transposeState = (false, true)
  var kernelDesc = GEMMKernelDescriptor(descriptor: gemmDesc)
  kernelDesc.device = nil
  kernelDesc.preferAsyncStore = nil
  let kernel = DerivativeSoftmaxKernel(
    descriptor: kernelDesc, D: D)
  
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
  
  // Generate the D[i] terms.
  var termsD = [Float](repeating: .zero, count: N)
  do {
    let O = network.inferenceAttention()
    
    // Iterate over the rows.
    for n in 0..<N {
      var dotProduct: Float = .zero
      for d in 0..<D {
        let matrixAddress = n * D + d
        dotProduct += O[matrixAddress] * network.C[matrixAddress]
      }
      termsD[n] = dotProduct
    }
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
  
  // Create Metal buffers from dO, V, and the D[i] terms.
  let bufferA = MTLContext.global
    .createBuffer(network.C, kernelDesc.memoryPrecisions!.A)
  let bufferB = MTLContext.global
    .createBuffer(network.V, kernelDesc.memoryPrecisions!.B)
  let bufferDTerms = MTLContext.global
    .createBuffer(termsD, .FP32)
  
  // Generate an initial value for P (until we advance to dS, where the actual
  // value will be used).
  var bufferC: MTLBuffer
  do {
    let inputMatrixP = matrixP // [Float](repeating: 1.0, count: N * N)
    var inputPrecisionP: GEMMOperandPrecision
    switch kernelDesc.memoryPrecisions!.C {
    case .FP32:
      inputPrecisionP = .FP32
    case .FP16:
      fatalError("Invalid precision for dP/dS.")
    case .BF16:
      inputPrecisionP = .FP16
    }
    bufferC = MTLContext.global
      .createBuffer(inputMatrixP, inputPrecisionP)
  }
  
  // Create the GEMM pipeline.
  let pipeline = DerivativeSoftmaxKernel.createPipeline(
    source: kernel.source, matrixDimensions: gemmDesc.matrixDimensions!)
  
  // Encode a single Metal command.
  do {
    let commandBuffer = MTLContext.global.commandQueue.makeCommandBuffer()!
    let encoder = commandBuffer.makeComputeCommandEncoder()!
    encoder.setComputePipelineState(pipeline)
    encoder.setThreadgroupMemoryLength(
      Int(kernel.threadgroupMemoryAllocation), index: 0)
    encoder.setBuffer(bufferA, offset: 0, index: 0)
    encoder.setBuffer(bufferB, offset: 0, index: 1)
    encoder.setBuffer(bufferC, offset: 0, index: 2)
    encoder.setBuffer(bufferDTerms, offset: 0, index: 3)
    
    func ceilDivide(_ target: Int, _ granularity: UInt16) -> Int {
      (target + Int(granularity) - 1) / Int(granularity)
    }
    let gridSize = MTLSize(
      width: ceilDivide(N, kernel.blockDimensions.N),
      height: ceilDivide(N, kernel.blockDimensions.M),
      depth: 1)
    let groupSize = MTLSize(
      width: Int(kernel.threadgroupSize),
      height: 1,
      depth: 1)
    encoder.dispatchThreadgroups(
      gridSize, threadsPerThreadgroup: groupSize)
    encoder.endEncoding()
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
  }
  
  // Copy the results.
  var result = [Float](repeating: .zero, count: N * N)
  do {
    let precision = kernelDesc.memoryPrecisions!.C
    let raw = bufferC.contents()
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
  
  // Check for correctness (not performance yet).
  print()
  print("result:")
  printMatrix(result)
  
  // Choose an error threshold.
  // - Only parametrized for (FP32, FP32, FP32) and (BF16, FP16, BF16).
  func createErrorThreshold(precision: GEMMOperandPrecision) -> Float {
    var empiricalMaxError: Float
    switch precision {
    case .FP32: empiricalMaxError = max(1.5e-7, Float(D).squareRoot() * 2.2e-8)
    case .FP16: fatalError("Accumulator cannot be FP16.")
    case .BF16: empiricalMaxError = 5e-3
    }
    return 2 * empiricalMaxError
  }
  
  // Check the results.
  var maxError: Float = .zero
  for r in 0..<N {
    for c in 0..<N {
      let address = r * N + c
      let expected = derivativeS[address]
      let actual = result[address]
      
      // Report whether it is correct.
      let error = (expected - actual).magnitude
      maxError = max(maxError, error)
    }
  }
  let errorThreshold = createErrorThreshold(
    precision: kernelDesc.memoryPrecisions!.C)
  if maxError > errorThreshold {
    print()
    print("max error: \(maxError) / ~1.000")
    print("Could not benchmark performance because results were incorrect.")
    return
  }
}
#endif
