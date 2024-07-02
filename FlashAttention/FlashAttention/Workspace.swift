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
  
  // Define the problem dimensions.
  let N: Int = 1000
  let D: Int = 56
  
  var networkDesc = NetworkDescriptor()
  networkDesc.N = N
  networkDesc.D = D
  var network = Network(descriptor: networkDesc)
  
  // Displays a matrix with dimensions N * 1.
  func printVector(_ matrix: [Float]) {
    for n in 0..<min(N, 10) {
      let matrixValue = matrix[n]
      var repr = String(format: "%.3f", matrixValue)
      while repr.count < 8 {
        repr = " " + repr
      }
      print(repr, terminator: " ")
    }
    print()
  }
  
  // Displays a matrix with dimensions N * D.
  func printMatrix(_ matrix: [Float]) {
    for d in 0..<min(D, 5) {
      for n in 0..<min(N, 10) {
        let matrixAddress = n * D + d
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
  
  // Displays a matrix with dimensions N * N.
  func printSquareMatrix(_ matrix: [Float]) {
    for rowID in 0..<min(N, 10) {
      for columnID in 0..<min(N, 10) {
        let matrixAddress = rowID * N + columnID
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
  print("Q:")
  printMatrix(network.Q)
  
  print()
  print("K:")
  printMatrix(network.K)
  
  print()
  print("S:")
  let S = (0..<N).flatMap(network.createMatrixSRow(rowID:))
  printSquareMatrix(S)
  
  print()
  print("P:")
  let P = (0..<N).flatMap(network.createMatrixPRow(rowID:))
  printSquareMatrix(P)
  
  print()
  print("LSE:")
  let LSE = (0..<N).map(network.createLSE(rowID:))
  printVector(LSE)
  
  print()
  print("V:")
  printMatrix(network.V)
  
  print()
  print("O:")
  printMatrix(network.inferenceAttention())
  
  var attentionDesc = AttentionDescriptor()
  attentionDesc.matrixDimensions = (R: UInt32(N), C: UInt32(N), D: UInt16(D))
  attentionDesc.memoryPrecisions = (Q: .full, K: .full, V: .full, O: .full)
  attentionDesc.transposeState = (Q: false, K: false, V: false, O: false)
  attentionDesc.type = .forward(true)
  let kernel = AttentionKernel(descriptor: attentionDesc)
  
  var pipeline: MTLComputePipelineState
  do {
    // Set the function constants.
    let constants = MTLFunctionConstantValues()
    var R = attentionDesc.matrixDimensions!.R
    var C = attentionDesc.matrixDimensions!.C
    var D = attentionDesc.matrixDimensions!.D
    constants.setConstantValue(&R, type: .uint, index: 0)
    constants.setConstantValue(&C, type: .uint, index: 1)
    constants.setConstantValue(&D, type: .ushort, index: 2)
    
    let device = MTLContext.global.device
    let library = try! device.makeLibrary(source: kernel.source, options: nil)
    let function = try! library.makeFunction(
      name: "attention", constantValues: constants)
    pipeline = try! device.makeComputePipelineState(function: function)
  }
  
  let bufferQ = MTLContext.global.createBuffer(network.Q, .FP32)
  let bufferK = MTLContext.global.createBuffer(network.K, .FP32)
  let bufferV = MTLContext.global.createBuffer(network.V, .FP32)
  
  var resultO = [Float](repeating: .zero, count: N * D)
  let bufferO = MTLContext.global.createBuffer(resultO, .FP32)
  
  var resultLSE = [Float](repeating: .zero, count: N)
  let bufferLSE = MTLContext.global.createBuffer(resultLSE, .FP32)
  
  do {
    let commandQueue = MTLContext.global.commandQueue
    let commandBuffer = commandQueue.makeCommandBuffer()!
    let encoder = commandBuffer.makeComputeCommandEncoder()!
    
    encoder.setComputePipelineState(pipeline)
    encoder.setThreadgroupMemoryLength(
      Int(kernel.threadgroupMemoryAllocation), index: 0)
    encoder.setBuffer(bufferQ, offset: 0, index: 0)
    encoder.setBuffer(bufferK, offset: 0, index: 1)
    encoder.setBuffer(bufferV, offset: 0, index: 2)
    encoder.setBuffer(bufferO, offset: 0, index: 3)
    encoder.setBuffer(bufferLSE, offset: 0, index: 4)
    
    do {
      func ceilDivide(_ target: Int, _ granularity: UInt16) -> Int {
        (target + Int(granularity) - 1) / Int(granularity)
      }
      let gridSize = MTLSize(
        width: ceilDivide(N, kernel.blockDimensions.R),
        height: 1,
        depth: 1)
      let groupSize = MTLSize(
        width: Int(kernel.threadgroupSize),
        height: 1,
        depth: 1)
      encoder.dispatchThreadgroups(
        gridSize, threadsPerThreadgroup: groupSize)
    }
    
    encoder.endEncoding()
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    
    // Determine the time taken.
    let start = commandBuffer.gpuStartTime
    let end = commandBuffer.gpuEndTime
    let latency = end - start
    let latencyMicroseconds = Int(latency / 1e-6)
    print(latencyMicroseconds, "μs")
  }
  
  // Copy the results.
  MTLContext.copy(bufferO, into: &resultO)
  MTLContext.copy(bufferLSE, into: &resultLSE)
  for i in resultLSE.indices {
    resultLSE[i] /= 1.44269504089
  }
  
  print()
  print("O:")
  printMatrix(resultO)
  
  print()
  print("LSE:")
  printVector(resultLSE)
  
  #if true
  // Check the results.
  let errorThreshold: Float = 1e-5
  var errorCount: Int = .zero
  let expectedO = network.inferenceAttention()
  for n in 0..<N {
    for d in 0..<D {
      let address = n * D + d
      let expected = expectedO[address]
      let actual = resultO[address]
      
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
  for n in 0..<N {
    let expected = LSE[n]
    let actual = resultLSE[n]
    
    // Report whether it is correct.
    let error = (expected - actual).magnitude
    if error > errorThreshold {
      if errorCount < 10 {
        print("error: \(error) / ~1.000")
        errorCount += 1
      }
    }
  }
  if errorCount > 0 {
    print("Could not benchmark performance because results were incorrect.")
    return
  }
  #endif
  
  // Move on to backward query.
}

#endif
