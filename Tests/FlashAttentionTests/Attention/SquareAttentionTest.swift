import XCTest
import FlashAttention

final class SquareAttentionTest: XCTestCase {
  func testCorrectness() throws {
    validateProblemSize(sequenceDimension: 8, headDimension: 8)
    validateProblemSize(sequenceDimension: 10, headDimension: 80)
    validateProblemSize(sequenceDimension: 8, headDimension: 2)
    validateProblemSize(sequenceDimension: 9, headDimension: 2)
    validateProblemSize(sequenceDimension: 23, headDimension: 2)
    validateProblemSize(sequenceDimension: 24, headDimension: 2)
    validateProblemSize(sequenceDimension: 25, headDimension: 2)
    validateProblemSize(sequenceDimension: 192, headDimension: 77)
    validateProblemSize(sequenceDimension: 192, headDimension: 80)
    validateProblemSize(sequenceDimension: 93, headDimension: 32)
    validateProblemSize(sequenceDimension: 99, headDimension: 35)
    validateProblemSize(sequenceDimension: 64, headDimension: 32)
    validateProblemSize(sequenceDimension: 64, headDimension: 34)
    validateProblemSize(sequenceDimension: 64, headDimension: 36)
    validateProblemSize(sequenceDimension: 64, headDimension: 40)
    validateProblemSize(sequenceDimension: 32, headDimension: 64)
    validateProblemSize(sequenceDimension: 4, headDimension: 1)
    validateProblemSize(sequenceDimension: 4, headDimension: 2)
    validateProblemSize(sequenceDimension: 384, headDimension: 95)
    validateProblemSize(sequenceDimension: 777, headDimension: 199)
  }
  
  func testPerformance() throws {
    // When searching a large combinatorial space, you probably want to
    // autogenerate the values for 'D' in this list. For example:
    //
    //    ```swift
    //    var D_array: [Int] = []
    //
    //    var D_cursor = 0
    //    while D_cursor < 96 {
    //      D_cursor += 4
    //      D_array.append(D_cursor)
    //    }
    //    while D_cursor < 160 {
    //      D_cursor += 8
    //      D_array.append(D_cursor)
    //    }
    //    while D_cursor < 256 {
    //      D_cursor += 16
    //      D_array.append(D_cursor)
    //    }
    //    while D_cursor < 384 {
    //      D_cursor += 32
    //      D_array.append(D_cursor)
    //    }
    //    ```
    //
    // If the test is extra expensive (e.g. testing 16384^2 instead of 8192^2,
    // make every 'D_cursor' increment jump twice as much. This change
    // reduces the test latency by half, compensating for the quadratic
    // scaling with increase of sequence length.
    //
    //    ```swift
    //    var D_array: [Int] = []
    //
    //    var D_cursor = 0
    //    while D_cursor < 96 {
    //      D_cursor += 8
    //      D_array.append(D_cursor)
    //    }
    //    while D_cursor < 160 {
    //      D_cursor += 16
    //      D_array.append(D_cursor)
    //    }
    //    while D_cursor < 256 {
    //      D_cursor += 32
    //      D_array.append(D_cursor)
    //    }
    //    while D_cursor < 384 {
    //      D_cursor += 64
    //      D_array.append(D_cursor)
    //    }
    //    ```
    //
    // The parameters for block size are separated into several logically
    // isolated zones of the D dimension spectrum. Often, you only need to
    // focus on a small subregion of this spectrum. To fix a single parameter
    // that is failing to perform better than the fallback / default.
    //
    // Testing only D = 4 to 64
    //
    //    ```swift
    //    var D_array: [Int] = []
    //
    //    var D_cursor = 0
    //    while D_cursor < 64 {
    //      D_cursor += 4
    //      D_array.append(D_cursor)
    //    }
    //    //while D_cursor < 160 {
    //    //  D_cursor += 8
    //    //  D_array.append(D_cursor)
    //    //}
    //    //while D_cursor < 256 {
    //    //  D_cursor += 16
    //    //  D_array.append(D_cursor)
    //    //}
    //    //while D_cursor < 384 {
    //    //  D_cursor += 32
    //    //  D_array.append(D_cursor)
    //    //}
    //    ```
    //
    // Testing only D = 128 to 256
    //
    //    ```swift
    //    var D_array: [Int] = []
    //
    //    var D_cursor = 128
    //    //while D_cursor < 96 {
    //    //  D_cursor += 4
    //    //  D_array.append(D_cursor)
    //    //}
    //    while D_cursor < 160 {
    //      D_cursor += 8
    //      D_array.append(D_cursor)
    //    }
    //    while D_cursor < 256 {
    //      D_cursor += 16
    //      D_array.append(D_cursor)
    //    }
    //    //while D_cursor < 384 {
    //    //  D_cursor += 32
    //    //  D_array.append(D_cursor)
    //    //}
    //    ```
    let D_array: [Int] = [16, 64, 256]
    
    // Sometimes, you may be focusing on a regression that impacts a single
    // kernel. Comment out the other two, to reduce latency. It may also
    // reduce the headache of sifting through three measurements, most of
    // which are not related to the answer your seek. I often made the mistake
    // of associating the 'FWD' metric with that of 'dK/dV'. My changes to the
    // code only affected backward performance, so the 'FWD' metrics would
    // always have the same value. I incorrectly concluded that the parameter
    // changes had no effect on 'dK/dV' performance.
    let kernelArray = [
      AttentionKernelType.forward,
      AttentionKernelType.backwardQuery,
      AttentionKernelType.backwardKeyValue,
    ]
    
    var outputString: String = ""
    print()
    
    // Loop over the configurations.
    for D in D_array {
      outputString += "\(D), "
      print("D =", D, terminator: ", ")
      
      for kernel in kernelArray {
        // N=1024 is too small to fully utilize the GPU. To actually benchmark
        // performance, use N=8192 (~M1 Max) or N=4096 (~M4). You need at least
        // 1024 threads per core and to saturate every core in the GPU. The
        // number of threads dispatched is 32 * (sequence length / 8).
        //
        // 32 * 8192 / 8 = 32768, 32768 >= 32 * 1024 (M1 Max)
        // 32 * 4096 / 8 = 16384, 16384 >= 10 * 1024 (M4)
        let metric = profileProblemSize(
          sequenceDimension: 1024,
          headDimension: D,
          benchmarkedKernel: kernel)
        outputString += "\(metric), "
        print(metric, terminator: ", ")
      }
      
      outputString.removeLast(2)
      outputString += "\n"
      print()
    }
    print()
    print(outputString)
  }
}

// Doesn't benchmark performance. Rather, it benchmarks correctness. The CPU
// code is too slow to have reasonable latencies for sequences in the range
// 4096 to 8192. Keep the sizes of test cases very small.
//
// This test is one of the reasons we need '-Xswiftc -Ounchecked'. It depends
// so critically on CPU-side performance. Here are some exemplary latencies.
// In a tight debugging feedback loop, you should target 1 second or less.
//
// ```
// swift test -Xswiftc -Ounchecked --filter SquareAttentionTest.testCorrectness
// Test Suite 'SquareAttentionTest' passed at 2024-08-23 10:37:53.604.
//   Executed 1 test, with 0 failures (0 unexpected) in 1.457 (1.457) seconds
// ```
//
// ```
// swift test --filter SquareAttentionTest.testCorrectness
// Test Suite 'SquareAttentionTest' passed at 2024-08-23 10:42:13.922.
//   Executed 1 test, with 0 failures (0 unexpected) in 28.969 (28.969) seconds
// ```
//
// ```
// swift test -Xswiftc -Onone --filter SquareAttentionTest.testCorrectness
// Same issues, skipped running the test to completion. Was taking too long and
// I was getting impatient.
// ```
//
// For the latter, I had to comment out 384 x 95 and 777 x 199, just to get a
// response in a reasonable amount of time. The actual latency would likely
// range in several minutes.
private func validateProblemSize(
  sequenceDimension: Int,
  headDimension: Int
) {
  var networkDesc = NetworkDescriptor()
  networkDesc.N = sequenceDimension
  networkDesc.D = headDimension
  let network = Network(descriptor: networkDesc)
  
  // MARK: - Kernels
  
  var attentionDesc = AttentionDescriptor()
  attentionDesc.lowPrecisionInputs = false
  attentionDesc.lowPrecisionIntermediates = false
  attentionDesc.matrixDimensions = (
    R: UInt32(sequenceDimension),
    C: UInt32(sequenceDimension),
    D: UInt16(headDimension))
  attentionDesc.transposeState = (Q: false, K: false, V: false, O: false)
  
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
    let memoryPrecisions = attentionDesc.memoryPrecisions()
    guard let precision = memoryPrecisions[operand] else {
      fatalError("Precision of operand \(operand) was not specified.")
    }
    return MTLContext.global.createBuffer(array, precision)
  }
  
  let operandSize = sequenceDimension * headDimension
  var resultO = [Float](repeating: .zero, count: operandSize)
  var resultL = [Float](repeating: .zero, count: sequenceDimension)
  var resultD = [Float](repeating: .zero, count: sequenceDimension)
  var resultDerivativeV = [Float](repeating: .zero, count: operandSize)
  var resultDerivativeK = [Float](repeating: .zero, count: operandSize)
  var resultDerivativeQ = [Float](repeating: .zero, count: operandSize)
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
        along: sequenceDimension)
      dispatch(
        kernel: kernelBackwardQuery,
        pipeline: pipelineBackwardQuery,
        along: sequenceDimension)
      dispatch(
        kernel: kernelBackwardKeyValue,
        pipeline: pipelineBackwardKeyValue,
        along: sequenceDimension)
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
    let memoryPrecisions = attentionDesc.memoryPrecisions()
    guard let precision = memoryPrecisions[operand] else {
      fatalError("Precision of operand \(operand) was not specified.")
    }
    MTLContext.copy(source, into: &destination, precision: precision)
  }
  
  let O = network.inferenceAttention()
  let L = (0..<sequenceDimension).map(network.createLTerm(rowID:))
  let D = (0..<sequenceDimension).map(network.createDTerm(rowID:))
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
      resultD[i] /= 1 / Float(headDimension).squareRoot()
    }
    
    copyBuffer(&resultO, bufferO, .O)
    copyBuffer(&resultDerivativeV, bufferDerivativeV, .dV)
    copyBuffer(&resultDerivativeK, bufferDerivativeK, .dK)
    copyBuffer(&resultDerivativeQ, bufferDerivativeQ, .dQ)
  }
  
  // This path floods the console a lot. Only activate it when debugging
  // correctness failures. Start by making the O matrix agree on both CPU
  // and GPU. Then, get the remaining operands to match.
#if false
  
  // Displays a matrix with dimensions N * 1.
  func printVector(_ matrix: [Float]) {
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
  func printMatrix(_ matrix: [Float]) {
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

// Returns throughput in gigainstructions per second.
private func profileProblemSize(
  sequenceDimension: Int,
  headDimension: Int,
  benchmarkedKernel: AttentionKernelType = .forward
) -> Int {
  var networkDesc = NetworkDescriptor()
  networkDesc.N = sequenceDimension
  networkDesc.D = headDimension
  let network = Network(descriptor: networkDesc)
  
  // MARK: - Kernels
  
  var attentionDesc = AttentionDescriptor()
  attentionDesc.lowPrecisionInputs = false
  attentionDesc.lowPrecisionIntermediates = false
  attentionDesc.matrixDimensions = (
    R: UInt32(sequenceDimension),
    C: UInt32(sequenceDimension),
    D: UInt16(headDimension))
  attentionDesc.transposeState = (Q: false, K: false, V: false, O: false)
  
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
    let memoryPrecisions = attentionDesc.memoryPrecisions()
    guard let precision = memoryPrecisions[operand] else {
      fatalError("Precision of operand \(operand) was not specified.")
    }
    return MTLContext.global.createBuffer(array, precision)
  }
  
  let operandSize = sequenceDimension * headDimension
  var resultO = [Float](repeating: .zero, count: operandSize)
  let resultL = [Float](repeating: .zero, count: sequenceDimension)
  let resultD = [Float](repeating: .zero, count: sequenceDimension)
  let resultDerivativeV = [Float](repeating: .zero, count: operandSize)
  let resultDerivativeK = [Float](repeating: .zero, count: operandSize)
  let resultDerivativeQ = [Float](repeating: .zero, count: operandSize)
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
      switch benchmarkedKernel {
      case .forward:
        dispatch(
          kernel: kernelForward,
          pipeline: pipelineForward,
          along: sequenceDimension)
      case .backwardQuery:
        dispatch(
          kernel: kernelBackwardQuery,
          pipeline: pipelineBackwardQuery,
          along: sequenceDimension)
      case .backwardKeyValue:
        dispatch(
          kernel: kernelBackwardKeyValue,
          pipeline: pipelineBackwardKeyValue,
          along: sequenceDimension)
      }
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
  
  // MARK: - Profiling
  
  // Benchmark performance.
  var maxGINSTRS: Int = .zero
  for _ in 0..<5 {
    let dispatchCount: Int = 5
    let latencySeconds = executeCommandBuffer(dispatchCount: dispatchCount)
    
    // Determine the amount of work done.
    //
    // WARNING: Change this code to match the kernel you're profiling.
    var operations: Int
    switch benchmarkedKernel {
    case .forward:
      operations = 2 * headDimension + 5
    case .backwardQuery:
      operations = 3 * headDimension + 5
    case .backwardKeyValue:
      operations = 4 * headDimension + 5
    }
    operations *= (sequenceDimension * sequenceDimension)
    operations *= dispatchCount
    
    // Divide the work by the latency, resulting in throughput.
    let instrs = Double(operations) / Double(latencySeconds)
    let ginstrs = Int(instrs / 1e9)
    
    // Accumulate the sample from this trial.
    maxGINSTRS = max(maxGINSTRS, ginstrs)
  }
  return maxGINSTRS
}
