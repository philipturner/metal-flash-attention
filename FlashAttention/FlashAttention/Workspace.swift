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
  //
  // 2560 - 490 GB/s (FP32, naive kernel)
  // 3072 - 537 GB/s (FP32, unoptimized custom kernel)
  //
  // 4096 - 399 GB/s (FP16, naive kernel)
  // 3072 - 270 GB/s (FP16, unoptimized custom kernel)
  let N: Int = 3072
  let D: Int = 3
  
  // Create the kernel.
  var softmaxDesc = SoftmaxDescriptor()
  
  // optimal size: 512-1024 (FP32, naive kernel)
  //               256-512 (FP16, naive kernel)
  softmaxDesc.threadgroupSize = 512
  softmaxDesc.memoryPrecision = .FP32
  softmaxDesc.matrixDimensions = (UInt16(N), UInt16(D))
  let softmaxKernel = SoftmaxKernel(descriptor: softmaxDesc)
  
  // Create the reference implementation.
  var networkDesc = NetworkDescriptor()
  networkDesc.N = N
  networkDesc.D = D
  let network = Network(descriptor: networkDesc)
  
  // Generate attention matrices with the reference implementation.
  var matrixS: [Float] = []
  var matrixP: [Float] = []
  for rowID in 0..<N {
    let matrixSRow = network.createMatrixSRow(rowID: rowID)
    let matrixPRow = network.createMatrixPRow(rowID: rowID)
    matrixS += matrixSRow
    matrixP += matrixPRow
  }
  
  // Create the pipeline state object.
  var pipeline: MTLComputePipelineState
  do {
    let library = try! MTLContext.global.device
      .makeLibrary(source: softmaxKernel.source, options: nil)
    let computeFunction = library.makeFunction(name: "softmax")!
    pipeline = try! MTLContext.global.device
      .makeComputePipelineState(function: computeFunction)
  }
  
  // MARK: - Correctness Test
  
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
  
  // Choose an error threshold.
  func createErrorThreshold(precision: GEMMOperandPrecision) -> Float {
    switch precision {
    case .FP32: return 1e-6
    case .FP16: return 1e-3
    case .BF16: return 1e-2
    }
  }
  
  // Check the results.
  let errorThreshold = createErrorThreshold(
    precision: softmaxDesc.memoryPrecision!)
  var errorCount: Int = .zero
  for r in 0..<N {
    for c in 0..<N {
      let address = r * N + c
      let expected = matrixP[address]
      let actual = result[address]
      
      // Report whether it is correct.
      let error = (expected - actual).magnitude
      if error > errorThreshold {
        if errorCount < 10 {
          print("error: \(error) / ~1.000")
          errorCount += 1
        }
      }
    }
  }
  if errorCount > 0 {
    print("Could not benchmark performance because results were incorrect.")
    return
  }
  
  // MARK: - Performance Test
  
  var maxBandwidth: Float = .zero
  var minLatency: Int = .max
  for _ in 0..<25 {
    let duplicatedCommandCount: Int = 50
    
    // Encode the GPU command.
    let commandBuffer = MTLContext.global.commandQueue.makeCommandBuffer()!
    let encoder = commandBuffer.makeComputeCommandEncoder()!
    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(attentionMatrixBuffer, offset: 0, index: 0)
    for _ in 0..<duplicatedCommandCount {
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
    
    // Determine the time taken.
    let start = commandBuffer.gpuStartTime
    let end = commandBuffer.gpuEndTime
    let latency = (end - start) / Double(duplicatedCommandCount)
    let latencyMicroseconds = Int(latency / 1e-6)
    
    // Determine the amount of work done.
    var bytes = 2 * N * N
    bytes *= softmaxDesc.memoryPrecision!.size
    let bandwidth = Float(bytes) / Float(latency) / 1e9
    
    // Accumulate the results.
    maxBandwidth = max(maxBandwidth, bandwidth)
    minLatency = min(minLatency, latencyMicroseconds)
  }
  
  // Report the results.
  func pad(_ string: String) -> String {
    var output = string
    while output.count < 8 {
      output = " " + output
    }
    return output
  }
  var problemSizeRepr = "\(N)"
  var latencyRepr = "\(minLatency)"
  var bandwidthRepr = String(format: "%.3f", maxBandwidth)
  problemSizeRepr = pad(problemSizeRepr)
  latencyRepr = pad(latencyRepr)
  bandwidthRepr = pad(bandwidthRepr)
  
  print("N = \(problemSizeRepr) | \(latencyRepr) μs | \(bandwidthRepr) GB/s")
}
#endif
