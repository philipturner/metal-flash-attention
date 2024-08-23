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
    descriptor.lowPrecisionInputs = false
    descriptor.lowPrecisionIntermediates = false
    descriptor.matrixDimensions = (row: 32, column: 13, head: 16)
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
  
  // MARK: - Transpose
  
  func transposeIn(_ input: [Float]) -> [Float] {
    let headDimension = Int(matrixDimensions.head)
    let sequenceDimension = input.count / headDimension
    
    var output = [Float](
      repeating: .zero, count: sequenceDimension * headDimension)
    for n in 0..<sequenceDimension {
      for d in 0..<headDimension {
        let inputAddress = n * headDimension + d
        let outputAddress = d * sequenceDimension + n
        output[outputAddress] = input[inputAddress]
      }
    }
    
    return output
  }
  
  func transposeOut(_ output: [Float]) -> [Float] {
    let headDimension = Int(matrixDimensions.head)
    let sequenceDimension = output.count / headDimension
    
    var input = [Float](
      repeating: .zero, count: sequenceDimension * headDimension)
    for n in 0..<sequenceDimension {
      for d in 0..<headDimension {
        let inputAddress = n * headDimension + d
        let outputAddress = d * sequenceDimension + n
        input[inputAddress] = output[outputAddress]
      }
    }
    
    return input
  }
  
  // Read the matrix inputs.
  var inputQ = network.Q
  var inputK = network.K
  var inputV = network.V
  var inputDerivativeO = network.dO
  if transposeState.Q {
    inputQ = transposeIn(inputQ)
  }
  if transposeState.K {
    inputK = transposeIn(inputK)
  }
  if transposeState.V {
    inputV = transposeIn(inputV)
  }
  if transposeState.O {
    inputDerivativeO = transposeIn(inputDerivativeO)
  }
  
  // MARK: - Buffers
  
  // Returns a zero-initialized array.
  func createArray(
    _ operand: AttentionOperand
  ) -> [Float] {
    var size: Int
    switch operand {
    case .K, .V, .dV, .dK:
      size = Int(matrixDimensions.column) * Int(matrixDimensions.head)
    case .Q, .O, .dO, .dQ:
      size = Int(matrixDimensions.row) * Int(matrixDimensions.head)
    case .L, .D:
      size = Int(matrixDimensions.row)
    default:
      fatalError("Unsupported operand.")
    }
    
    return [Float](repeating: .zero, count: size)
  }
  
  // Returns a buffer.
  func createBuffer(
    _ operand: AttentionOperand,
    contents: [Float]
  ) -> MTLBuffer {
    let memoryPrecisions = attentionDesc.memoryPrecisions
    guard let precision = memoryPrecisions[operand] else {
      fatalError("Precision of operand \(operand) was not specified.")
    }
    return MTLContext.global.createBuffer(contents, precision)
  }
  
  // Write the matrix inputs.
  let bufferQ = createBuffer(.Q, contents: inputQ)
  let bufferK = createBuffer(.K, contents: inputK)
  let bufferV = createBuffer(.V, contents: inputV)
  let bufferDerivativeO = createBuffer(.dO, contents: inputDerivativeO)
  
  // Allocate the per-row intermediates.
  let bufferL = createBuffer(.L, contents: createArray(.L))
  let bufferD = createBuffer(.D, contents: createArray(.D))
  
  // Allocate the matrix outputs.
  let bufferO = createBuffer(.O, contents: createArray(.O))
  let bufferDerivativeV = createBuffer(.dV, contents: createArray(.dV))
  let bufferDerivativeK = createBuffer(.dK, contents: createArray(.dK))
  let bufferDerivativeQ = createBuffer(.dQ, contents: createArray(.dQ))
  
  // MARK: - Issuing GPU Work
  
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
  
  // MARK: - Collecting Results
  
  // Read a buffer.
  func readBuffer(
    _ operand: AttentionOperand,
    contents: MTLBuffer
  ) -> [Float] {
    let memoryPrecisions = attentionDesc.memoryPrecisions
    guard let precision = memoryPrecisions[operand] else {
      fatalError("Precision of operand \(operand) was not specified.")
    }
    
    var destination = createArray(operand)
    MTLContext.copy(contents, into: &destination, precision: precision)
    return destination
  }
  
  // Read the per-row intermediates.
  var resultL = readBuffer(.L, contents: bufferL)
  var resultD = readBuffer(.D, contents: bufferD)
  for i in resultL.indices {
    resultL[i] /= 1.44269504089
  }
  for i in resultD.indices {
    resultD[i] /= 1 / Float(matrixDimensions.head).squareRoot()
  }
  
  // Read the matrix outputs.
  var resultO = readBuffer(.O, contents: bufferO)
  var resultDerivativeV = readBuffer(.dV, contents: bufferDerivativeV)
  var resultDerivativeK = readBuffer(.dK, contents: bufferDerivativeK)
  var resultDerivativeQ = readBuffer(.dQ, contents: bufferDerivativeQ)
  if transposeState.Q {
    resultDerivativeQ = transposeOut(resultDerivativeQ)
  }
  if transposeState.K {
    resultDerivativeK = transposeOut(resultDerivativeK)
  }
  if transposeState.V {
    resultDerivativeV = transposeOut(resultDerivativeV)
  }
  if transposeState.O {
    resultO = transposeOut(resultO)
  }
  
  // MARK: - Validation
  
  // Query the expected outputs on the reference implementation.
  let O = network.inferenceAttention()
  let L = (0..<Int(matrixDimensions.row)).map(network.createLTerm(rowID:))
  let D = (0..<Int(matrixDimensions.row)).map(network.createDTerm(rowID:))
  let dV = network.derivativeV()
  let dK = network.derivativeK()
  let dQ = network.derivativeQ()
  
  // This path floods the console a lot. Only activate it when debugging
  // correctness failures. Start by making the O matrix agree on both CPU
  // and GPU. Then, get the remaining operands to match.
#if false
  
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
