import XCTest
import FlashAttention

final class RectangularAttentionTest: XCTestCase {
  // Tests random permutations of transpose state and input/output sequence
  // length. Just like the old MFA test suite.
  //
  // For simplicity, we are only testing FP32. This removes the need to worry
  // about numerical rounding error. With mixed precision, the rounding
  // error scales with problem dimension in predictable ways. We have
  // discovered predictive formulae for GEMM through trial and error.
  func testCorrectness() throws {
    var descriptor = AttentionDescriptor()
    descriptor.lowPrecisionInputs = true
    descriptor.lowPrecisionIntermediates = true
    descriptor.matrixDimensions = (row: 12, column: 23, head: 8)
    descriptor.transposeState = (Q: false, K: false, V: false, O: false)
    runCorrectnessTest(descriptor: descriptor)
  }
}

/// Run a test with the specified configuration.
private func runCorrectnessTest(descriptor: AttentionDescriptor) {
  // Check that all properties of the descriptor have been set.
  guard let matrixDimensions = descriptor.matrixDimensions,
        let transposeState = descriptor.transposeState else {
    fatalError("Descriptor was incomplete.")
  }
//  guard !descriptor.lowPrecisionInputs,
//        !descriptor.lowPrecisionIntermediates else {
//    fatalError("Mixed precision is not supported.")
//  }
  
  var networkDesc = NetworkDescriptor()
  networkDesc.rowDimension = Int(matrixDimensions.row)
  networkDesc.columnDimension = Int(matrixDimensions.column)
  networkDesc.headDimension = Int(matrixDimensions.head)
  let network = Network(descriptor: networkDesc)
  
  // MARK: - Kernels
  
  let attentionDesc = descriptor
  
  func createKernel(type: AttentionKernelType) -> AttentionKernel {
    let attentionKernelDesc = attentionDesc.kernelDescriptor(type: type)
    let attentionKernel = AttentionKernel(descriptor: attentionKernelDesc)
    return attentionKernel
  }
  let kernelForward = createKernel(type: .forward)
  let kernelBackwardQuery = createKernel(type: .backwardQuery)
  let kernelBackwardKeyValue = createKernel(type: .backwardKeyValue)
  
  func createPipeline(kernel: AttentionKernel) -> MTLComputePipelineState {
    let device = MTLContext.global.device
    let source = kernel.createSource()
    let library = try! device.makeLibrary(source: source, options: nil)
    
    let functionConstants = MTLFunctionConstantValues()
    attentionDesc.setFunctionConstants(functionConstants)
    let function = try! library.makeFunction(
      name: "attention", constantValues: functionConstants)
    
    // A critical part of the heuristic: force the occupancy to 1024 on M1.
    let pipelineDesc = MTLComputePipelineDescriptor()
    pipelineDesc.computeFunction = function
    pipelineDesc.maxTotalThreadsPerThreadgroup = 1024
    return try! device.makeComputePipelineState(
      descriptor: pipelineDesc, options: [], reflection: nil)
  }
  let pipelineForward = createPipeline(kernel: kernelForward)
  let pipelineBackwardQuery = createPipeline(kernel: kernelBackwardQuery)
  let pipelineBackwardKeyValue = createPipeline(kernel: kernelBackwardKeyValue)
  
  // MARK: - Buffers
  
  // Utility function to make buffer initialization more concise.
  func createBuffer(
    _ array: [Float],
    _ operand: AttentionOperand
  ) -> MTLBuffer {
    let memoryPrecisions = attentionDesc.memoryPrecisions
    guard let precision = memoryPrecisions[operand] else {
      fatalError("Precision of operand \(operand) was not specified.")
    }
    return MTLContext.global.createBuffer(array, precision)
  }
  
  var resultO = [Float](
    repeating: .zero,
    count: Int(matrixDimensions.row) * Int(matrixDimensions.head))
  var resultL = [Float](
    repeating: .zero,
    count: Int(matrixDimensions.row))
  var resultD = [Float](
    repeating: .zero,
    count: Int(matrixDimensions.row))
  var resultDerivativeV = [Float](
    repeating: .zero, 
    count: Int(matrixDimensions.column) * Int(matrixDimensions.head))
  var resultDerivativeK = [Float](
    repeating: .zero,
    count: Int(matrixDimensions.column) * Int(matrixDimensions.head))
  var resultDerivativeQ = [Float](
    repeating: .zero,
    count: Int(matrixDimensions.row) * Int(matrixDimensions.head))
  resultO[0] = .nan
  
  let bufferQ = createBuffer(network.Q, .Q)
  let bufferK = createBuffer(network.K, .K)
  let bufferV = createBuffer(network.V, .V)
  let bufferDerivativeO = createBuffer(network.dO, .dO)
  
  let bufferL = createBuffer(resultL, .L)
  let bufferD = createBuffer(resultD, .D)
  
  let bufferO = createBuffer(resultO, .O)
  let bufferDerivativeV = createBuffer(resultDerivativeV, .dV)
  let bufferDerivativeK = createBuffer(resultDerivativeK, .dK)
  let bufferDerivativeQ = createBuffer(resultDerivativeQ, .dQ)
  
  // MARK: - GPU Commands
  
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
      along parallelizationDimension: Int
    ) {
      encoder.setComputePipelineState(pipeline)
      encoder.setThreadgroupMemoryLength(
        Int(kernel.threadgroupMemoryAllocation), index: 0)
      
      let blockCount = ceilDivide(
        parallelizationDimension, kernel.blockDimensions.parallelization)
      let gridSize = MTLSize(
        width: blockCount,
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
    
    encoder.setBuffer(bufferL, offset: 0, index: 4)
    encoder.setBuffer(bufferD, offset: 0, index: 5)
    
    encoder.setBuffer(bufferDerivativeO, offset: 0, index: 6)
    encoder.setBuffer(bufferDerivativeV, offset: 0, index: 7)
    encoder.setBuffer(bufferDerivativeK, offset: 0, index: 8)
    encoder.setBuffer(bufferDerivativeQ, offset: 0, index: 9)
    
    for _ in 0..<dispatchCount {
      dispatch(
        kernel: kernelForward,
        pipeline: pipelineForward,
        along: Int(matrixDimensions.row))
      dispatch(
        kernel: kernelBackwardQuery,
        pipeline: pipelineBackwardQuery,
        along: Int(matrixDimensions.row))
      dispatch(
        kernel: kernelBackwardKeyValue,
        pipeline: pipelineBackwardKeyValue,
        along: Int(matrixDimensions.column))
    }
    
    encoder.endEncoding()
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    
    // Determine the time taken.
    let start = commandBuffer.gpuStartTime
    let end = commandBuffer.gpuEndTime
    let latency = end - start
    return latency
  }
  executeCommandBuffer(dispatchCount: 1)
  
  // MARK: - Validation
  
  // Utility function to make buffer copying more concise.
  func copyBuffer(
    _ destination: inout [Float],
    _ source: MTLBuffer,
    _ operand: AttentionOperand
  ) {
    let memoryPrecisions = attentionDesc.memoryPrecisions
    guard let precision = memoryPrecisions[operand] else {
      fatalError("Precision of operand \(operand) was not specified.")
    }
    MTLContext.copy(source, into: &destination, precision: precision)
  }
  
  let O = network.inferenceAttention()
  let L = (0..<Int(matrixDimensions.row)).map(network.createLTerm(rowID:))
  let D = (0..<Int(matrixDimensions.row)).map(network.createDTerm(rowID:))
  let dV = network.derivativeV()
  let dK = network.derivativeK()
  let dQ = network.derivativeQ()
  
  // Copy the results.
  do {
    copyBuffer(&resultL, bufferL, .L)
    copyBuffer(&resultD, bufferD, .D)
    for i in resultL.indices {
      resultL[i] /= 1.44269504089
    }
    for i in resultD.indices {
      resultD[i] /= 1 / Float(matrixDimensions.head).squareRoot()
    }
    
    copyBuffer(&resultO, bufferO, .O)
    copyBuffer(&resultDerivativeV, bufferDerivativeV, .dV)
    copyBuffer(&resultDerivativeK, bufferDerivativeK, .dK)
    copyBuffer(&resultDerivativeQ, bufferDerivativeQ, .dQ)
  }
  
  // This path floods the console a lot. Only activate it when debugging
  // correctness failures. Start by making the O matrix agree on both CPU
  // and GPU. Then, get the remaining operands to match.
#if true
  
  // Displays a matrix with dimensions N * 1.
  func printVector(_ matrix: [Float]) {
    let sequenceDimension = matrix.count / 1
    
    for n in 0..<min(sequenceDimension, 10) {
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
  //
  // The entered matrix cannot be transposed.
  func printMatrix(_ matrix: [Float]) {
    let sequenceDimension = matrix.count / Int(matrixDimensions.head)
    let headDimension = Int(matrixDimensions.head)
    
    for d in 0..<min(headDimension, 5) {
      for n in 0..<min(sequenceDimension, 10) {
        let matrixAddress = n * headDimension + d
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
  print("L:")
  printVector(L)
  
  print()
  print("L:")
  printVector(resultL)
  
  print()
  print("D:")
  printVector(D)
  
  print()
  print("D:")
  printVector(resultD)
  
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
  
  var errorCount: Int = .zero
  func check(expected: [Float], actual: [Float], tolerance: Float) {
    guard expected.count == actual.count else {
      fatalError("Arrays had different length.")
    }
    
    for i in expected.indices {
      let error = (expected[i] - actual[i]).magnitude
      if error > tolerance || error.isNaN {
        // Don't report errors in this case.
        if (expected[i].isNaN || expected[i].isInfinite),
           (actual[i].isNaN || actual[i].isInfinite ) {
          continue
        }
        
        // Update the error count in the outer scope.
        if errorCount < 10 {
          errorCount += 1
          print("error: \(error) / ~1.000")
          print("- expected[\(i)] =", expected[i])
          print("-   actual[\(i)] =", actual[i])
        }
      }
    }
  }
  
  // Check the results.
  if attentionDesc.lowPrecisionInputs ||
      attentionDesc.lowPrecisionIntermediates {
    check(expected: O, actual: resultO, tolerance: 5e-2)
    check(expected: L, actual: resultL, tolerance: 7e-3)
    check(expected: D, actual: resultD, tolerance: 1e-1)
    check(expected: dV, actual: resultDerivativeV, tolerance: 5e-2)
    check(expected: dK, actual: resultDerivativeK, tolerance: 5e-2)
    check(expected: dQ, actual: resultDerivativeQ, tolerance: 5e-2)
  } else {
    check(expected: O, actual: resultO, tolerance: 2e-5)
    check(expected: L, actual: resultL, tolerance: 2e-5)
    check(expected: D, actual: resultD, tolerance: 2e-5)
    check(expected: dV, actual: resultDerivativeV, tolerance: 2e-5)
    check(expected: dK, actual: resultDerivativeK, tolerance: 2e-5)
    check(expected: dQ, actual: resultDerivativeQ, tolerance: 2e-5)
  }
}
