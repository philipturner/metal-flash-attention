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
  print("Hello, console.")
  
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
  
  // Set up a correctness test with matrix dimensions typical for attention.
  let N: Int = 2000
  let D: Int = 100
  
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
  
  // MARK: - Correctness Test
  
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
  
  // MARK: - Performance Test
  
  // Test the GEMM throughput on M1 and M4.
  //
  // M1 Max, FP32:
  //
  // N = 1496 | D =   48 |       74 μs | 3000 GFLOPS
  // N = 1498 | D =   48 |       77 μs | 2881 GFLOPS
  // N = 1500 | D =   48 |       76 μs | 2940 GFLOPS
  // N = 1502 | D =   48 |       77 μs | 2911 GFLOPS
  // N = 1504 | D =   48 |       71 μs | 3154 GFLOPS
  //
  // N = 3000 | D =   48 |      165 μs | 5451 GFLOPS
  // N = 3000 | D =   50 |      319 μs | 2928 GFLOPS
  // N = 3000 | D =   52 |      270 μs | 3595 GFLOPS
  // N = 3000 | D =   54 |      237 μs | 4236 GFLOPS
  // N = 3000 | D =   56 |      185 μs | 5631 GFLOPS
  //
  // N = 3000 | D =   96 |      262 μs | 6707 GFLOPS
  // N = 3000 | D =   98 |      429 μs | 4188 GFLOPS
  // N = 3000 | D =  100 |      370 μs | 4961 GFLOPS
  // N = 3000 | D =  102 |      334 μs | 5595 GFLOPS
  // N = 3000 | D =  104 |      281 μs | 6783 GFLOPS
  //
  // N = 3000 | D =  248 |      572 μs | 7865 GFLOPS
  // N = 3000 | D =  250 |      754 μs | 6015 GFLOPS
  // N = 3000 | D =  252 |      689 μs | 6628 GFLOPS
  // N = 3000 | D =  254 |      661 μs | 6969 GFLOPS
  // N = 3000 | D =  256 |      599 μs | 7744 GFLOPS
  //
  // M1 Max, BF16 and FP16:
  //
  // N = 1496 | D =   48 |      123 μs | 1811 GFLOPS
  // N = 1498 | D =   48 |      124 μs | 1798 GFLOPS
  // N = 1500 | D =   48 |      124 μs | 1804 GFLOPS
  // N = 1502 | D =   48 |      124 μs | 1808 GFLOPS
  // N = 1504 | D =   48 |      122 μs | 1853 GFLOPS
  //
  // N = 3000 | D =   48 |      129 μs | 6975 GFLOPS
  // N = 3000 | D =   50 |      235 μs | 3975 GFLOPS
  // N = 3000 | D =   52 |      195 μs | 4961 GFLOPS
  // N = 3000 | D =   54 |      159 μs | 6325 GFLOPS
  // N = 3000 | D =   56 |      170 μs | 6139 GFLOPS
  //
  // N = 3000 | D =   96 |      223 μs | 7884 GFLOPS
  // N = 3000 | D =  100 |      279 μs | 6563 GFLOPS
  // N = 3000 | D =  104 |      247 μs | 7723 GFLOPS
  //
  // N = 3000 | D =  248 |      536 μs | 8393 GFLOPS
  // N = 3000 | D =  250 |      588 μs | 7703 GFLOPS
  // N = 3000 | D =  256 |      551 μs | 8426 GFLOPS
  //
  // M4, FP32:
  //
  // N = 1000 | D =   40 |      142 μs |  589 GFLOPS
  // N = 1000 | D =   42 |       78 μs | 1126 GFLOPS
  // N = 1000 | D =   44 |      117 μs |  781 GFLOPS
  // N = 1000 | D =   46 |      119 μs |  800 GFLOPS
  // N = 1000 | D =   48 |      120 μs |  827 GFLOPS
  // N = 1000 | D =   50 |       87 μs | 1184 GFLOPS
  // N = 1000 | D =   52 |       86 μs | 1243 GFLOPS
  // N = 1000 | D =   54 |       88 μs | 1267 GFLOPS
  // N = 1000 | D =   56 |       81 μs | 1424 GFLOPS
  // N = 1000 | D =   58 |       96 μs | 1244 GFLOPS
  // N = 1000 | D =   60 |       95 μs | 1303 GFLOPS
  //
  // N = 1496 | D =   48 |      143 μs | 1556 GFLOPS
  // N = 1498 | D =   48 |      178 μs | 1260 GFLOPS
  // N = 1500 | D =   48 |      155 μs | 1446 GFLOPS
  // N = 1502 | D =   48 |      161 μs | 1393 GFLOPS
  // N = 1504 | D =   48 |      145 μs | 1553 GFLOPS
  //
  // N = 2000 | D =   48 |      328 μs | 1219 GFLOPS
  // N = 2000 | D =   50 |      334 μs | 1242 GFLOPS
  // N = 2000 | D =   52 |      329 μs | 1310 GFLOPS
  // N = 2000 | D =   54 |      334 μs | 1338 GFLOPS
  // N = 2000 | D =   56 |      335 μs | 1382 GFLOPS
  // N = 2000 | D =   96 |      351 μs | 2232 GFLOPS
  // N = 2000 | D =  100 |      376 μs | 2169 GFLOPS
  // N = 2000 | D =  104 |      369 μs | 2292 GFLOPS
  //
  // M4, BF16 and FP16:
  //
  // N = 1000 | D =   40 |      152 μs |  551 GFLOPS
  // N = 1000 | D =   48 |       95 μs | 1043 GFLOPS
  // N = 1000 | D =   56 |       71 μs | 1615 GFLOPS
  // N = 1000 | D =   60 |       80 μs | 1541 GFLOPS
  //
  // N = 1496 | D =   48 |      129 μs | 1732 GFLOPS
  // N = 1498 | D =   48 |      131 μs | 1703 GFLOPS
  // N = 1500 | D =   48 |      128 μs | 1754 GFLOPS
  // N = 1502 | D =   48 |      114 μs | 1968 GFLOPS
  // N = 1504 | D =   48 |      126 μs | 1788 GFLOPS
  //
  // N = 2000 | D =   48 |      140 μs | 2855 GFLOPS
  // N = 2000 | D =   50 |      161 μs | 2573 GFLOPS
  // N = 2000 | D =   52 |      161 μs | 2676 GFLOPS
  // N = 2000 | D =   54 |      163 μs | 2741 GFLOPS
  // N = 2000 | D =   56 |      158 μs | 2932 GFLOPS
  // N = 2000 | D =   96 |      249 μs | 3140 GFLOPS
  // N = 2000 | D =  100 |      271 μs | 3004 GFLOPS
  // N = 2000 | D =  104 |      267 μs | 3165 GFLOPS
  
  var maxGFLOPS: Int = .zero
  var minLatency: Int = .max
  for _ in 0..<10 {
    let duplicatedCommandCount: Int = 10
    
    // Encode the GPU command.
    let commandBuffer = MTLContext.global.commandQueue.makeCommandBuffer()!
    let encoder = commandBuffer.makeComputeCommandEncoder()!
    encoder.setComputePipelineState(pipeline)
    encoder.setThreadgroupMemoryLength(
      Int(kernel.threadgroupMemoryAllocation), index: 0)
    encoder.setBuffer(bufferA, offset: 0, index: 0)
    encoder.setBuffer(bufferB, offset: 0, index: 1)
    encoder.setBuffer(bufferC, offset: 0, index: 2)
    encoder.setBuffer(bufferDTerms, offset: 0, index: 3)
    
    for _ in 0..<duplicatedCommandCount {
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
    let instructions = (1 + D + 1) * N * N
    let operations = 2 * instructions
    let gflops = Int(Double(operations) / Double(latency) / 1e9)
    
    // Accumulate the results.
    maxGFLOPS = max(maxGFLOPS, gflops)
    minLatency = min(minLatency, latencyMicroseconds)
  }
  
  // Report the results.
  func pad(_ string: String, length: Int) -> String {
    var output = string
    while output.count < length {
      output = " " + output
    }
    return output
  }
  
  var problemSizeRepr = "\(N)"
  var headSizeRepr = "\(D)"
  var latencyRepr = "\(minLatency)"
  var gflopsRepr = "\(maxGFLOPS)"
  problemSizeRepr = pad(problemSizeRepr, length: 4)
  headSizeRepr = pad(headSizeRepr, length: 4)
  latencyRepr = pad(latencyRepr, length: 8)
  gflopsRepr = pad(gflopsRepr, length: 4)
  
  print()
  print("N = \(problemSizeRepr)", terminator: " | ")
  print("D = \(headSizeRepr)", terminator: " | ")
  print("\(latencyRepr) μs", terminator: " | ")
  print("\(gflopsRepr) GFLOPS")
}
#endif
