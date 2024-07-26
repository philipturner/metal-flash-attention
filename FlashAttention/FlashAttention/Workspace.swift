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
  // Automate the execution of the test suite.
  profileProblemSize(N: 10, D: 3)
  profileProblemSize(N: 10, D: 80)
  profileProblemSize(N: 8, D: 2)
  profileProblemSize(N: 9, D: 2)
  profileProblemSize(N: 23, D: 2)
  profileProblemSize(N: 24, D: 2)
  profileProblemSize(N: 25, D: 2)
  profileProblemSize(N: 192, D: 77)
  profileProblemSize(N: 192, D: 80)
  profileProblemSize(N: 93, D: 32)
  profileProblemSize(N: 99, D: 35)
  profileProblemSize(N: 64, D: 32)
  profileProblemSize(N: 32, D: 64)
  profileProblemSize(N: 4, D: 1)
  profileProblemSize(N: 4, D: 2)
  profileProblemSize(N: 384, D: 95)
  profileProblemSize(N: 777, D: 199)
  
//  let N_array = [128]
//  let D_array = [32, 48, 64, 80, 96, 128, 160, 192, 256]
//  var outputString: String = ""
//  for N in N_array {
//    print("N =", N, terminator: ", ")
//    outputString += "N = \(N), "
//    for D in D_array {
//      let ginstrs = profileProblemSize(N: N, D: D)
//      print(ginstrs, terminator: ", ")
//      outputString += "\(ginstrs), "
//    }
//    outputString.removeLast(2)
//    print()
//    outputString += "\n"
//  }
//  print()
//  print(outputString)
  
  /*
   Before - all applicable variables cached
   
   latency: 61
   latency: 135
   latency: 62
   latency: 64
   latency: 64
   latency: 61
   latency: 64
   latency: 522
   latency: 561
   latency: 145
   latency: 217
   latency: 115
   latency: 113
   latency: 51
   latency: 49
   latency: 1150
   latency: 6763
   
   Before - all applicable variables paged
   
   latency: 61
   latency: 137
   latency: 58
   latency: 67
   latency: 62
   latency: 63
   latency: 65
   latency: 698
   latency: 701
   latency: 164
   latency: 261
   latency: 119
   latency: 119
   latency: 48
   latency: 47
   latency: 1477
   latency: 7272
   
   Original kernel - V full, others disable
   
   latency: 56
   latency: 121
   latency: 51
   latency: 52
   latency: 62
   latency: 60
   latency: 61
   latency: 713
   latency: 751
   latency: 161
   latency: 323
   latency: 119
   latency: 114
   latency: 47
   latency: 51
   latency: 1507
   latency: 7150
   
   Original kernel - V disable, others disable
   
   latency: 54
   latency: 160
   latency: 51
   latency: 53
   latency: 63
   latency: 60
   latency: 64
   latency: 872
   latency: 840
   latency: 158
   latency: 380
   latency: 129
   latency: 153
   latency: 45
   latency: 51
   latency: 1869
   latency: 8206
   
   */
}

// Returns: Throughput in GINSTRS.
@discardableResult
func profileProblemSize(N: Int, D: Int) -> Int {
  // Remaining optimizations:
  // - Try an explicit register spilling mode, where async copies are used to
  //   minimize the overhead of paging. Use the output buffers as the scratch
  //   space.
  //   - Postponing the application of this optimization to dV. I didn't see a
  //     major speedup for the other accumulators yet. I need to understand why
  //     performance did not improve.
  //   - TODO: Do not unroll outer-product loops when the respective LHS is
  //     intentionally spilled.
  //   - TODO: Do not unroll accumulate loops when the respective output is
  //     intentionally spilled.
  // - Insight: Register spilling is closely related to "store dS^T" with a set
  //   of accumulators and temporary materialization of an attention submatrix.
  // - Elide async copies on M3. Can the R edge (FWD, BWD dQ) and C edge
  //   (BWD dK/dV) be masked out through alternative means?
  
  var networkDesc = NetworkDescriptor()
  networkDesc.N = N
  networkDesc.D = D
  let network = Network(descriptor: networkDesc)
  
  var attentionDesc = AttentionDescriptor()
  attentionDesc.cachedInputs = (Q: false, K: false, V: false, dO: false)
  attentionDesc.cachedOutputs = (dQ: false, dK: false, dV: true, O: false)
  attentionDesc.matrixDimensions = (R: UInt32(N), C: UInt32(N), D: UInt16(D))
  attentionDesc.transposeState = (Q: false, K: false, V: false, O: false)
  
  attentionDesc.type = .forward(true)
  let kernelForward = AttentionKernel(descriptor: attentionDesc)
  
  attentionDesc.type = .backwardQuery(true)
  let kernelBackwardQuery = AttentionKernel(descriptor: attentionDesc)
  
  attentionDesc.type = .backwardKeyValue(true)
  let kernelBackwardKeyValue = AttentionKernel(descriptor: attentionDesc)
  
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
  let pipelineBackwardKeyValue = createPipeline(kernel: kernelBackwardKeyValue)
  
  let bufferQ = MTLContext.global.createBuffer(network.Q, .FP32)
  let bufferK = MTLContext.global.createBuffer(network.K, .FP32)
  let bufferV = MTLContext.global.createBuffer(network.V, .FP32)
  let bufferDerivativeO = MTLContext.global.createBuffer(network.C, .FP32)
  
  var resultO = [Float](repeating: .zero, count: N * D)
  resultO[0] = .nan
  var resultLTerms = [Float](repeating: .zero, count: N)
  var resultDTerms = [Float](repeating: .zero, count: N)
  var resultDerivativeV = [Float](repeating: .zero, count: N * D)
  var resultDerivativeK = [Float](repeating: .zero, count: N * D)
  var resultDerivativeQ = [Float](repeating: .zero, count: N * D)
  
  let bufferO = MTLContext.global.createBuffer(resultO, .FP32)
  let bufferLTerms = MTLContext.global.createBuffer(resultLTerms, .FP32)
  let bufferDTerms = MTLContext.global.createBuffer(resultDTerms, .FP32)
  let bufferDerivativeV = MTLContext.global
    .createBuffer(resultDerivativeV, .FP32)
  let bufferDerivativeK = MTLContext.global
    .createBuffer(resultDerivativeK, .FP32)
  let bufferDerivativeQ = MTLContext.global
    .createBuffer(resultDerivativeQ, .FP32)
  
  // - Parameter dispatchCount: Number of times to duplicate the FWD / BWD
  //                            combined pass.
  // - Returns: Latency of the entire command buffer, in seconds.
  @discardableResult
  func executeCommandBuffer(
    dispatchCount: Int
  ) -> Double {
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
    
    encoder.setBuffer(bufferQ, offset: 0, index: 0)
    encoder.setBuffer(bufferK, offset: 0, index: 1)
    encoder.setBuffer(bufferV, offset: 0, index: 2)
    encoder.setBuffer(bufferO, offset: 0, index: 3)
    encoder.setBuffer(bufferLTerms, offset: 0, index: 4)
    
    encoder.setBuffer(bufferDerivativeO, offset: 0, index: 5)
    encoder.setBuffer(bufferDTerms, offset: 0, index: 6)
    encoder.setBuffer(bufferDerivativeV, offset: 0, index: 7)
    encoder.setBuffer(bufferDerivativeK, offset: 0, index: 8)
    encoder.setBuffer(bufferDerivativeQ, offset: 0, index: 9)
    
    for _ in 0..<dispatchCount {
      if dispatchCount > 1 {
        // WARNING: Change this code to match the kernel you're profiling.
        dispatch(
          kernel: kernelForward,
          pipeline: pipelineForward,
          along: kernelForward.blockDimensions.R)
      } else {
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
      }
    }
    
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
  executeCommandBuffer(dispatchCount: 1)
  
  #if true
  let O = network.inferenceAttention()
  let LTerms = (0..<N).map(network.createLTerm(rowID:))
  let DTerms = (0..<N).map(network.createDTerm(rowID:))
  let dV = network.derivativeV()
  let dK = network.derivativeK()
  let dQ = network.derivativeQ()
  
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
  MTLContext.copy(bufferDerivativeK, into: &resultDerivativeK)
  MTLContext.copy(bufferDerivativeQ, into: &resultDerivativeQ)
  
  #if false
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
  
  print()
  print("dQ:")
  printMatrix(dQ)
  
  print()
  print("dQ:")
  printMatrix(resultDerivativeQ)
  #endif
  
  // Check the results.
  //
  // Error thresholds:
  // - Everything in FP32: 1e-5
  // - Testing the "Store dS" variant with dS in BF16: 1e-2
  let errorThreshold: Float = 1e-2
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
  check(expected: dK, actual: resultDerivativeK)
  check(expected: dQ, actual: resultDerivativeQ)
  #endif
  
  #if false
  // Benchmark performance.
  var maxGINSTRS: Int = .zero
  for _ in 0..<5 {
    let dispatchCount: Int = 5
    let latencySeconds = executeCommandBuffer(dispatchCount: dispatchCount)
    
    // Determine the amount of work done.
    //
    // WARNING: Change this code to match the kernel you're profiling.
    var operations: Int = .zero
    operations += (2 * D + 5) * (N * N)
    operations *= dispatchCount
    
    // Divide the work by the latency, resulting in throughput.
    let instrs = Double(operations) / Double(latencySeconds)
    let ginstrs = Int(instrs / 1e9)
    
    // Accumulate the sample from this trial.
    maxGINSTRS = max(maxGINSTRS, ginstrs)
  }
  return maxGINSTRS
  #endif
  
  // WARNING: Change this to match the kernel being profiled.
  return pipelineBackwardKeyValue.maxTotalThreadsPerThreadgroup
}
