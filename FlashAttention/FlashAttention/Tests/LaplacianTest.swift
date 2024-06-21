//
//  LaplacianTest.swift
//  FlashAttention
//
//  Created by Philip Turner on 6/21/24.
//

import Metal

#if false

func executeScript() {
  print("Hello, console.")
  
  struct TestDescriptor {
    var precision: GEMMOperandPrecision?
    var problemSize: Int?
    var transposeState: (Bool, Bool)?
  }
  func runTest(descriptor: TestDescriptor) {
    guard let precision = descriptor.precision,
          let problemSize = descriptor.problemSize,
          let transposeState = descriptor.transposeState else {
      fatalError("Descriptor was incomplete.")
    }
    
    // Set up the kernel.
    var gemmDesc = GEMMDescriptor()
    let n = UInt32(problemSize)
    gemmDesc.matrixDimensions = (M: n, N: n, K: n)
    gemmDesc.memoryPrecisions = (precision, precision, precision)
    gemmDesc.transposeState = descriptor.transposeState
    
    // Test the kernel.
    let statistic = profileProblemSize(descriptor: gemmDesc)
    
    // Report the results.
    do {
      var repr = "\(problemSize)"
      while repr.count < 4 {
        repr = " " + repr
      }
      print("problemSize = \(repr)", terminator: " | ")
    }
    if transposeState.0 {
      print("A^T", terminator: " ")
    } else {
      print("A  ", terminator: " ")
    }
    if transposeState.1 {
      print("B^T", terminator: " | ")
    } else {
      print("B  ", terminator: " | ")
    }
    
    for laneID in [Int(1), Int(0)] {
      var repr = "\(statistic[laneID])"
      while repr.count < 4 {
        repr = " " + repr
      }
      
      // Log the number to the console.
      if laneID == 0 {
        print(repr, terminator: " GFLOPS")
      } else {
        print(repr, terminator: " threads/core | ")
      }
    }
    print("")
  }
  
  // Correctness tests.
  do {
    let problemSizes: [Int] = [
      7, 8, 9, 10,
      15, 16, 17, 18,
      23, 24, 25,
      31, 32, 33,
      47, 48, 49,
      63, 64, 65,
      103, 104, 112,
      126, 127, 128, 129,
      130, 131,
      135, 136, 137,
      143, 144, 145,
      151, 152, 153,
    ]
    let transposeStates: [(Bool, Bool)] = [
      (false, false),
      (false, true),
      (true, false),
    ]
    
    print()
    print("Correctness tests:")
    for problemSize in problemSizes {
      for transposeState in transposeStates {
        var testDescriptor = TestDescriptor()
        testDescriptor.precision = .FP32
        testDescriptor.problemSize = problemSize
        testDescriptor.transposeState = transposeState
        runTest(descriptor: testDescriptor)
      }
    }
  }
  
  // My workspace. Edit this code to run actual tests.
  do {
    let transposeStates: [(Bool, Bool)] = [
      (false, false),
      (false, true),
      (true, false),
      (true, true),
    ]
    
    // Working on investigating BF16 performance with large matrices.
    print()
    print("Performance tests:")
    for problemSize in 511...512 {
      for transposeState in transposeStates {
        var testDescriptor = TestDescriptor()
        testDescriptor.precision = .BF16
        testDescriptor.problemSize = problemSize
        testDescriptor.transposeState = transposeState
        runTest(descriptor: testDescriptor)
      }
    }
  }
}

/// A continuous (integration) test of both correctness and performance. This
/// test completes with low latency (\<1 second) for rapid feedback during
/// iterative design.
///
/// Returns:
/// - lane 0: maximum achieved performance in GFLOPS
/// - lane 1: occupancy in threads/core
func profileProblemSize(
  descriptor: GEMMDescriptor
) -> SIMD2<Int> {
  guard let matrixDimensions = descriptor.matrixDimensions,
        matrixDimensions.M == matrixDimensions.N,
        matrixDimensions.M == matrixDimensions.K else {
    fatalError("Matrix dimensions were invalid.")
  }
  let problemSize = Int(matrixDimensions.M)
  
  // Allocate FP32 memory for the operands.
  var A = [Float](repeating: .zero, count: problemSize * problemSize)
  var B = [Float](repeating: .zero, count: problemSize * problemSize)
  var C = [Float](repeating: .zero, count: problemSize * problemSize)
  
  // Initialize A as the 2nd-order periodic Laplacian.
  for diagonalID in 0..<problemSize {
    let diagonalAddress = diagonalID * problemSize + diagonalID
    A[diagonalAddress] = -2
    
    let leftColumnID = (diagonalID + problemSize - 1) % problemSize
    let leftSubDiagonalAddress = diagonalID * problemSize + leftColumnID
    A[leftSubDiagonalAddress] = 1
    
    let rightColumnID = (diagonalID + problemSize + 1) % problemSize
    let rightSubDiagonalAddress = diagonalID * problemSize + rightColumnID
    A[rightSubDiagonalAddress] = 1
  }
  
  // Initialize B to random numbers.
  for rowID in 0..<problemSize {
    for columnID in 0..<problemSize {
      let address = rowID * problemSize + columnID
      let entry = Float.random(in: 0..<1)
      B[address] = entry
    }
  }
  
  // Since the Laplacian is symmetric, we swap roles of the matrices to test
  // transposition of the left-hand side.
  //
  // Note that the test cannot cover correctness of A and B transposition
  // simultaneously. Instead, test the correctness in isolation
  // (AB, AB^T, A^T B). Performance can be tested in all four permutations
  // (AB, AB^T, A^T B, A^T B^T).
  if descriptor.transposeState!.A {
    swap(&A, &B)
  }
  
  // Initialize the context.
  let device = MTLCreateSystemDefaultDevice()!
  let commandQueue = device.makeCommandQueue()!
  let context = (device: device, commandQueue: commandQueue)
  
  func createBuffer(
    _ originalData: [Float],
    _ precision: GEMMOperandPrecision
  ) -> MTLBuffer {
    // Add random numbers to expose out-of-bounds accesses.
    var augmentedData = originalData
    for _ in 0..<originalData.count {
      let randomNumber = Float.random(in: -2...2)
      augmentedData.append(randomNumber)
    }
    
    // Allocate enough memory to store everything in Float32.
    let bufferSize = augmentedData.count * 4
    let buffer = context.device.makeBuffer(length: bufferSize)!
    
    // Copy the data into the buffer.
    switch precision {
    case .FP32:
      let pointer = buffer.contents().assumingMemoryBound(to: Float.self)
      for i in augmentedData.indices {
        pointer[i] = augmentedData[i]
      }
    case .FP16:
      let pointer = buffer.contents().assumingMemoryBound(to: Float16.self)
      for i in augmentedData.indices {
        pointer[i] = Float16(augmentedData[i])
      }
    case .BF16:
      let pointer = buffer.contents().assumingMemoryBound(to: UInt16.self)
      for i in augmentedData.indices {
        let value32 = augmentedData[i].bitPattern
        let value16 = unsafeBitCast(value32, to: SIMD2<UInt16>.self)[1]
        pointer[i] = value16
      }
    }
    return buffer
  }
  
  // Multiply A with B.
  var maxGFLOPS: Int = .zero
  var occupancy: Int = .zero
  do {
    // Generate the kernel.
    let (kernel, pipeline) = retrieveGEMMKernel(descriptor: descriptor)
    occupancy = pipeline.maxTotalThreadsPerThreadgroup
    
    // Create the buffers.
    let bufferA = createBuffer(A, descriptor.memoryPrecisions!.A)
    let bufferB = createBuffer(B, descriptor.memoryPrecisions!.B)
    let bufferC = createBuffer(C, descriptor.memoryPrecisions!.C)
    
    // Profile the latency of matrix multiplication.
    for _ in 0..<15 {
      let duplicatedCommandCount: Int = 20
      
      // Encode the GPU command.
      let commandBuffer = context.commandQueue.makeCommandBuffer()!
      let encoder = commandBuffer.makeComputeCommandEncoder()!
      encoder.setComputePipelineState(pipeline)
      encoder.setThreadgroupMemoryLength(
        Int(kernel.threadgroupMemoryAllocation), index: 0)
      encoder.setBuffer(bufferA, offset: 0, index: 0)
      encoder.setBuffer(bufferB, offset: 0, index: 1)
      encoder.setBuffer(bufferC, offset: 0, index: 2)
      for _ in 0..<duplicatedCommandCount {
        func ceilDivide(_ target: Int, _ granularity: UInt16) -> Int {
          (target + Int(granularity) - 1) / Int(granularity)
        }
        let gridSize = MTLSize(
          width: ceilDivide(problemSize, kernel.blockDimensions.N),
          height: ceilDivide(problemSize, kernel.blockDimensions.M),
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
      
      // Determine the amount of work done.
      var operations = 2 * problemSize * problemSize * problemSize
      operations = operations * duplicatedCommandCount
      let gflops = Int(Double(operations) / Double(latency) / 1e9)
      
      // Report the results.
      // let latencyMicroseconds = Int(latency / 1e-6)
      // print(latencyMicroseconds, "μs", gflops, "GFLOPS")
      maxGFLOPS = max(maxGFLOPS, gflops)
    }
    
    // Copy the results to C.
    do {
      let precision = descriptor.memoryPrecisions!.C
      let raw = bufferC.contents()
      for rowID in 0..<problemSize {
        for columnID in 0..<problemSize {
          let address = rowID * problemSize + columnID
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
          C[address] = entry32
        }
      }
    }
  }
  
  // Choose an error threshold.
  func createErrorThreshold(precision: GEMMOperandPrecision) -> Float {
    switch precision {
    case .FP32: return 1e-5
    case .FP16: return 5e-3
    case .BF16: return 5e-2
    }
  }
  var errorThreshold: Float = 0
  do {
    let memoryPrecisions = descriptor.memoryPrecisions!
    let thresholdA = createErrorThreshold(precision: memoryPrecisions.A)
    let thresholdB = createErrorThreshold(precision: memoryPrecisions.B)
    let thresholdC = createErrorThreshold(precision: memoryPrecisions.C)
    errorThreshold = max(errorThreshold, thresholdA)
    errorThreshold = max(errorThreshold, thresholdB)
    errorThreshold = max(errorThreshold, thresholdC)
  }
  
  // Check the results.
  var errorCount: Int = .zero
  for m in 0..<problemSize {
    for n in 0..<problemSize {
      // Find the source row IDs.
      let leftRowID = (m + problemSize - 1) % problemSize
      let centerRowID = m
      let rightRowID = (m + problemSize + 1) % problemSize
      
      // Find the source scalars.
      var leftSource: Float
      var centerSource: Float
      var rightSource: Float
      if descriptor.transposeState!.A {
        leftSource = A[leftRowID * problemSize + n]
        centerSource = A[centerRowID * problemSize + n]
        rightSource = A[rightRowID * problemSize + n]
      } else if descriptor.transposeState!.B {
        leftSource = B[n * problemSize + leftRowID]
        centerSource = B[n * problemSize + centerRowID]
        rightSource = B[n * problemSize + rightRowID]
      } else {
        leftSource = B[leftRowID * problemSize + n]
        centerSource = B[centerRowID * problemSize + n]
        rightSource = B[rightRowID * problemSize + n]
      }
      
      // Find the expected result.
      let expected = leftSource - 2 * centerSource + rightSource
      
      // Find the actual result.
      var actual: Float
      if descriptor.transposeState!.A {
        actual = C[n * problemSize + m]
      } else {
        actual = C[m * problemSize + n]
      }
      
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
  return SIMD2(maxGFLOPS, occupancy)
}
#endif
