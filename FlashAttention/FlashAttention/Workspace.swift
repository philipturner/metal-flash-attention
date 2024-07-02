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
  let N: Int = 100
  let D: Int = 200
  
  var networkDesc = NetworkDescriptor()
  networkDesc.N = N
  networkDesc.D = D
  let network = Network(descriptor: networkDesc)
  
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
  print("L_terms:")
  let LTerms = (0..<N).map {
    network.createLTerm(rowID: $0) * 1.44269504089
  }
  printVector(LTerms)
  
  print()
  print("D_terms:")
  let DTerms = (0..<N).map {
    network.createDTerm(rowID: $0)
  }
  printVector(DTerms)
  
  print()
  print("O:")
  printMatrix(network.inferenceAttention())
  
  print()
  print("dO:")
  printMatrix(network.C)
  
  var attentionDesc = AttentionDescriptor()
  attentionDesc.matrixDimensions = (R: UInt32(N), C: UInt32(N), D: UInt16(D))
  attentionDesc.memoryPrecisions = (Q: .full, K: .full, V: .full, O: .full)
  attentionDesc.transposeState = (Q: false, K: false, V: false, O: false)
  
  attentionDesc.type = .forward(true)
  let kernelForward = AttentionKernel(descriptor: attentionDesc)
  
  attentionDesc.type = .backwardQuery(false)
  let kernelBackwardQuery = AttentionKernel(descriptor: attentionDesc)
  
  func createPipeline(kernel: AttentionKernel) -> MTLComputePipelineState {
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
    return try! device.makeComputePipelineState(function: function)
  }
  let pipelineForward = createPipeline(kernel: kernelForward)
  let pipelineBackwardQuery = createPipeline(kernel: kernelBackwardQuery)
  
  let bufferQ = MTLContext.global.createBuffer(network.Q, .FP32)
  let bufferK = MTLContext.global.createBuffer(network.K, .FP32)
  let bufferV = MTLContext.global.createBuffer(network.V, .FP32)
  let bufferDerivativeO = MTLContext.global.createBuffer(network.C, .FP32)
  
  var resultO = [Float](repeating: .zero, count: N * D)
  var resultLTerms = [Float](repeating: .zero, count: N)
  var resultDTerms = [Float](repeating: .zero, count: N)
  
  let bufferO = MTLContext.global.createBuffer(resultO, .FP32)
  let bufferLTerms = MTLContext.global.createBuffer(resultLTerms, .FP32)
  let bufferDTerms = MTLContext.global.createBuffer(resultDTerms, .FP32)
  
  do {
    let commandQueue = MTLContext.global.commandQueue
    let commandBuffer = commandQueue.makeCommandBuffer()!
    let encoder = commandBuffer.makeComputeCommandEncoder()!
    
    encoder.setBuffer(bufferQ, offset: 0, index: 0)
    encoder.setBuffer(bufferK, offset: 0, index: 1)
    encoder.setBuffer(bufferV, offset: 0, index: 2)
    encoder.setBuffer(bufferO, offset: 0, index: 3)
    encoder.setBuffer(bufferLTerms, offset: 0, index: 4)
    
    encoder.setBuffer(bufferDerivativeO, offset: 0, index: 5)
    encoder.setBuffer(bufferDTerms, offset: 0, index: 6)
    
    func ceilDivide(_ target: Int, _ granularity: UInt16) -> Int {
      (target + Int(granularity) - 1) / Int(granularity)
    }
    func dispatch(kernel: AttentionKernel, pipeline: MTLComputePipelineState) {
      encoder.setComputePipelineState(pipeline)
      encoder.setThreadgroupMemoryLength(
        Int(kernel.threadgroupMemoryAllocation), index: 0)
      
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
    
    dispatch(
      kernel: kernelForward, 
      pipeline: pipelineForward)
    dispatch(
      kernel: kernelBackwardQuery,
      pipeline: pipelineBackwardQuery)
    
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
  MTLContext.copy(bufferLTerms, into: &resultLTerms)
  MTLContext.copy(bufferDTerms, into: &resultDTerms)
  
  print()
  print("O:")
  printMatrix(resultO)
  
  print()
  print("L_terms:")
  printVector(resultLTerms)
  
  print()
  print("D_terms:")
  printVector(resultDTerms)
  
  #if true
  // Check the results.
  let errorThreshold: Float = 1e-5
  var errorCount: Int = .zero
  func check(expected: [Float], actual: [Float]) {
    guard expected.count == actual.count else {
      fatalError("Arrays had different length.")
    }
    
    for i in expected.indices {
      let error = (expected[i] - actual[i]).magnitude
      if error > errorThreshold {
        if errorCount < 10 {
          // Update the error count in the outer scope.
          errorCount += 1
          print("error: \(error) / ~1.000")
        }
      }
    }
  }
  
  let expectedO = network.inferenceAttention()
  check(expected: expectedO, actual: resultO)
  check(expected: LTerms, actual: resultLTerms)
  check(expected: DTerms, actual: resultDTerms)
  if errorCount > 0 {
    print("Could not benchmark performance because results were incorrect.")
    return
  }
  #endif
  
  
}

#endif
