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
  //
  // Tasks:
  // - Support the attention variant that stores dS^T to memory.
  // - Compare performance with FP32 and BF16 storage.
  
  #if false
  // Define the problem dimensions.
  let N: Int = 128
  
  var samples: [Int] = []
  for D in [32, 64, 128, 192] {
    let performance = profileProblemSize(N: N, D: D)
    samples.append(performance)
  }
  
  print()
  for sample in samples {
    print(sample, terminator: ", ")
  }
  print()
  #else
  
  // The bug happens whenever D < N.
  _ = profileProblemSize(N: 64, D: 32)
  #endif
}

func profileProblemSize(N: Int, D: Int) -> Int {
  //
  // Correctness tests:
  //
  // Test N=10, D=3/80
  // Test N=8/9/24/25, D=2
  // Test N=192, D=80
  
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
    var matrixDimensionFloat = Double(matrix.count)
    matrixDimensionFloat.formSquareRoot()
    matrixDimensionFloat.round(.toNearestOrEven)
    let matrixDimensionInt = Int(matrixDimensionFloat)
    guard matrixDimensionInt * matrixDimensionInt == matrix.count else {
      fatalError("Unable to take square root of integer.")
    }
    
    for rowID in 0..<min(matrixDimensionInt, 10) {
      for columnID in 0..<min(matrixDimensionInt, 10) {
        let matrixAddress = rowID * matrixDimensionInt + columnID
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
  
#if true
  // Display the attention matrices.
  do {
    print()
    print("S:")
    let S = (0..<N).flatMap(network.createMatrixSRow(rowID:))
    printSquareMatrix(S)
    
    print()
    print("P:")
    let P = (0..<N).flatMap(network.createMatrixPRow(rowID:))
    printSquareMatrix(P)
    
    print()
    print("dP:")
    let dP = (0..<N).flatMap(network.createDerivativePRow(rowID:))
    printSquareMatrix(dP)
    
    print()
    print("dS:")
    let dS = (0..<N).flatMap(network.createDerivativeSRow(rowID:))
    printSquareMatrix(dS)
  }
  
  let O = network.inferenceAttention()
  let LTerms = (0..<N).map(network.createLTerm(rowID:))
  let DTerms = (0..<N).map(network.createDTerm(rowID:))
  let dV = network.derivativeV()
  let dK = network.derivativeK()
#endif
  
  var attentionDesc = AttentionDescriptor()
  attentionDesc.matrixDimensions = (R: UInt32(N), C: UInt32(N), D: UInt16(D))
  attentionDesc.memoryPrecisions = (Q: .full, K: .full, V: .full, O: .full)
  attentionDesc.transposeState = (Q: false, K: false, V: false, O: false)
  
  attentionDesc.type = .forward(true)
  let kernelForward = AttentionKernel(descriptor: attentionDesc)
  
  attentionDesc.type = .backwardQuery(false)
  let kernelBackwardQuery = AttentionKernel(descriptor: attentionDesc)
  
  attentionDesc.type = .backwardKeyValue(false)
  let kernelBackwardKeyValue = AttentionKernel(descriptor: attentionDesc)
  
  // dK = dS^T Q
  var gemmDerivativeK: GEMMKernel
  do {
    // MxNxK (BLAS notation) <-> NxDxN (Attention notation)
    var gemmDesc = GEMMDescriptor()
    gemmDesc.matrixDimensions = (M: UInt32(N), N: UInt32(D), K: UInt32(N))
    gemmDesc.memoryPrecisions = (A: .FP32, B: .FP32, C: .FP32)
    gemmDesc.transposeState = (A: false, B: false)
    
    var gemmKernelDesc = GEMMKernelDescriptor(descriptor: gemmDesc)
    gemmKernelDesc.device = MTLContext.global.device
    gemmKernelDesc.leadingDimensions = (
      "\(kernelBackwardKeyValue.leadingDimensionDerivativeST)", "N")
    gemmKernelDesc.preferAsyncStore = true
    gemmDerivativeK = GEMMKernel(descriptor: gemmKernelDesc)
  }
  
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
  func createPipeline(library: MTLLibrary) -> MTLComputePipelineState {
    // Set the function constants.
    let constants = MTLFunctionConstantValues()
    var M = UInt32(N)
    var N = UInt32(D)
    var K = UInt32(N)
    constants.setConstantValue(&M, type: .uint, index: 0)
    constants.setConstantValue(&N, type: .uint, index: 1)
    constants.setConstantValue(&K, type: .uint, index: 2)
    
    let device = MTLContext.global.device
    let function = try! library.makeFunction(
      name: "gemm", constantValues: constants)
    let pipeline = try! device.makeComputePipelineState(function: function)
    return pipeline
  }
  let pipelineForward = createPipeline(kernel: kernelForward)
  let pipelineBackwardQuery = createPipeline(kernel: kernelBackwardQuery)
  let pipelineBackwardKeyValue = createPipeline(kernel: kernelBackwardKeyValue)
  let pipelineDerivativeK = createPipeline(library: gemmDerivativeK.library)
  
  let bufferQ = MTLContext.global.createBuffer(network.Q, .FP32)
  let bufferK = MTLContext.global.createBuffer(network.K, .FP32)
  let bufferV = MTLContext.global.createBuffer(network.V, .FP32)
  let bufferDerivativeO = MTLContext.global.createBuffer(network.C, .FP32)
  
  var resultO = [Float](repeating: .zero, count: N * D)
  var resultLTerms = [Float](repeating: .zero, count: N)
  var resultDTerms = [Float](repeating: .zero, count: N)
  var resultDerivativeV = [Float](repeating: .zero, count: N * D)
  let bufferO = MTLContext.global.createBuffer(resultO, .FP32)
  let bufferLTerms = MTLContext.global.createBuffer(resultLTerms, .FP32)
  let bufferDTerms = MTLContext.global.createBuffer(resultDTerms, .FP32)
  let bufferDerivativeV = MTLContext.global
    .createBuffer(resultDerivativeV, .FP32)
  
  let N_padded = Int(kernelBackwardKeyValue.leadingDimensionDerivativeST)
  var resultDerivativeST = [Float](repeating: .zero, count: N_padded * N_padded)
  var resultDerivativeK = [Float](repeating: .zero, count: N * D)
  let bufferDerivativeST = MTLContext.global
    .createBuffer(resultDerivativeST, .FP32)
  let bufferDerivativeK = MTLContext.global
    .createBuffer(resultDerivativeK, .FP32)
  
  // - Parameter dispatchCount: Number of times to duplicate the FWD / BWD
  //                            combined pass.
  // - Returns: Latency of the entire command buffer, in seconds.
  @discardableResult
  func executeCommandBuffer() -> Double {
    let commandQueue = MTLContext.global.commandQueue
    let commandBuffer = commandQueue.makeCommandBuffer()!
    let encoder = commandBuffer.makeComputeCommandEncoder()!
    
    func ceilDivide(_ target: Int, _ granularity: UInt16) -> Int {
      (target + Int(granularity) - 1) / Int(granularity)
    }
    
    // Bind all necessary MTLBuffer arguments before calling this function.
    func dispatch(
      kernel: AttentionKernel,
      pipeline: MTLComputePipelineState,
      along matrixSide: UInt16 // left (R/rows), top (C/columns)
    ) {
      encoder.setComputePipelineState(pipeline)
      encoder.setThreadgroupMemoryLength(
        Int(kernel.threadgroupMemoryAllocation), index: 0)
      
      let gridSize = MTLSize(
        width: ceilDivide(N, matrixSide),
        height: 1,
        depth: 1)
      let groupSize = MTLSize(
        width: Int(kernel.threadgroupSize),
        height: 1,
        depth: 1)
      encoder.dispatchThreadgroups(
        gridSize, threadsPerThreadgroup: groupSize)
    }
    
    // Bind all necessary MTLBuffer arguments before calling this function.
    func dispatch(
      kernel: GEMMKernel,
      pipeline: MTLComputePipelineState
    ) {
      encoder.setComputePipelineState(pipeline)
      encoder.setThreadgroupMemoryLength(
        Int(kernel.threadgroupMemoryAllocation), index: 0)
      
      let gridSize = MTLSize(
        width: ceilDivide(D, kernel.blockDimensions.N),
        height: ceilDivide(N, kernel.blockDimensions.M),
        depth: 1)
      let groupSize = MTLSize(
        width: Int(kernel.threadgroupSize),
        height: 1,
        depth: 1)
      encoder.dispatchThreadgroups(
        gridSize, threadsPerThreadgroup: groupSize)
    }
    
    encoder.setBuffer(bufferQ, offset: 0, index: 0)
    encoder.setBuffer(bufferK, offset: 0, index: 1)
    encoder.setBuffer(bufferV, offset: 0, index: 2)
    encoder.setBuffer(bufferO, offset: 0, index: 3)
    encoder.setBuffer(bufferLTerms, offset: 0, index: 4)
    
    encoder.setBuffer(bufferDerivativeO, offset: 0, index: 5)
    encoder.setBuffer(bufferDTerms, offset: 0, index: 6)
    encoder.setBuffer(bufferDerivativeV, offset: 0, index: 7)
    encoder.setBuffer(bufferDerivativeST, offset: 0, index: 8)
    dispatch(
      kernel: kernelForward,
      pipeline: pipelineForward,
      along: kernelForward.blockDimensions.R)
    dispatch(
      kernel: kernelBackwardQuery,
      pipeline: pipelineBackwardQuery,
      along: kernelBackwardQuery.blockDimensions.R)
    dispatch(
      kernel: kernelBackwardKeyValue,
      pipeline: pipelineBackwardKeyValue,
      along: kernelBackwardKeyValue.blockDimensions.C)
    
    encoder.setBuffer(bufferDerivativeST, offset: 0, index: 0)
    encoder.setBuffer(bufferQ, offset: 0, index: 1)
    encoder.setBuffer(bufferDerivativeK, offset: 0, index: 2)
    dispatch(
      kernel: gemmDerivativeK,
      pipeline: pipelineDerivativeK)
    
    encoder.endEncoding()
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    
    // Determine the time taken.
    let start = commandBuffer.gpuStartTime
    let end = commandBuffer.gpuEndTime
    let latency = end - start
    print("latency:", Int(latency * 1e6))
    return latency
  }
  executeCommandBuffer()
  
  // Copy the results.
  MTLContext.copy(bufferO, into: &resultO)
  MTLContext.copy(bufferLTerms, into: &resultLTerms)
  MTLContext.copy(bufferDTerms, into: &resultDTerms)
  for i in resultLTerms.indices {
    resultLTerms[i] /= 1.44269504089
  }
  for i in resultDTerms.indices {
    resultDTerms[i] /= 1 / Float(D).squareRoot()
  }
  MTLContext.copy(bufferDerivativeV, into: &resultDerivativeV)
  MTLContext.copy(
    bufferDerivativeST, into: &resultDerivativeST, precision: .FP32)
  MTLContext.copy(bufferDerivativeK, into: &resultDerivativeK)
  
#if true
  print()
  print("dST:")
  printSquareMatrix(resultDerivativeST)
  
  print()
  print("Q:")
  printMatrix(network.Q)
  
  print()
  print("V:")
  printMatrix(network.V)
  
  print()
  print("O:")
  printMatrix(O)
  
  print()
  print("O:")
  printMatrix(resultO)
  
  print()
  print("L_terms:")
  printVector(LTerms)
  
  print()
  print("L_terms:")
  printVector(resultLTerms)
  
  print()
  print("D_terms:")
  printVector(DTerms)
  
  print()
  print("D_terms:")
  printVector(resultDTerms)
  
  print()
  print("dV:")
  printMatrix(dV)
  
  print()
  print("dV:")
  printMatrix(resultDerivativeV)
  
  print()
  print("dK:")
  printMatrix(dK)
  
  print()
  print("dK:")
  printMatrix(resultDerivativeK)
#endif
  
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
      if error > errorThreshold || error.isNaN {
        if errorCount < 10 {
          // Update the error count in the outer scope.
          errorCount += 1
          print("error: \(error) / ~1.000")
          print("- expected[\(i)] =", expected[i])
          print("-   actual[\(i)] =", actual[i])
        }
      }
    }
  }
  
  check(expected: O, actual: resultO)
  check(expected: LTerms, actual: resultLTerms)
  check(expected: DTerms, actual: resultDTerms)
  check(expected: dV, actual: resultDerivativeV)
  if errorCount > 0 {
    print("Could not benchmark performance because results were incorrect.")
    return 0
  }
#endif
  return 0
  
  #if false
  // Benchmark performance.
  print()
  var maxGINSTRS: Int = .zero
  for _ in 0..<5 {
    let dispatchCount: Int = 5
    let latencySeconds = executeCommandBuffer(dispatchCount: dispatchCount)
    let latencyMicroseconds = Int(latencySeconds * 1e6)
    
    // Determine the amount of work done.
    var operations: Int = .zero
    operations += (2 * D + 5) * (N * N) // forward pass
    operations += (5 * D + 5) * (N * N) // backward pass
    operations *= dispatchCount
    
    // Divide the work by the latency, resulting in throughput.
    let intrs = Double(operations) / Double(latencySeconds)
    let gintrs = Int(intrs / 1e9)
    print(gintrs, "GINSTRS", "-", latencyMicroseconds, "μs")
    
    // Accumulate the sample from this trial.
    maxGINSTRS = max(maxGINSTRS, gintrs)
  }
  return maxGINSTRS
  #endif
}

#endif
