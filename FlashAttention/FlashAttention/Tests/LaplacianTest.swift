//
//  LaplacianTest.swift
//  FlashAttention
//
//  Created by Philip Turner on 6/21/24.
//

import Metal

// Tests the performance of large matrix multiplications, using the
// second-order Laplacian in direct matrix form.

#if true
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
    gemmDesc.loadPreviousC = false
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
    for problemSize in 1488...1489 {
      for transposeState in transposeStates {
        var testDescriptor = TestDescriptor()
        testDescriptor.precision = .BF16
        testDescriptor.problemSize = problemSize
        testDescriptor.transposeState = transposeState
        runTest(descriptor: testDescriptor)
      }
    }
    
    print()
    print("Performance tests:")
    for problemSize in 1488...1489 {
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
  var previousOperandC = [Float](
    repeating: .zero, count: problemSize * problemSize)
  
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
  
  // Initialize C to random numbers.
  for rowID in 0..<problemSize {
    for columnID in 0..<problemSize {
      let address = rowID * problemSize + columnID
      let entry = Float.random(in: 0..<1)
      previousOperandC[address] = entry
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
  
  // Multiply A with B.
  var maxGFLOPS: Int = .zero
  var occupancy: Int = .zero
  do {
    // Generate the kernel.
    GEMMKernel.register(descriptor: descriptor)
    let (kernel, pipeline) = GEMMKernel.pipelineCache[descriptor]!
    occupancy = pipeline.maxTotalThreadsPerThreadgroup
    
    // Create the buffers.
    let bufferA = MTLContext.global
      .createBuffer(A, descriptor.memoryPrecisions!.A)
    let bufferB = MTLContext.global
      .createBuffer(B, descriptor.memoryPrecisions!.B)
    let bufferC = MTLContext.global
      .createBuffer(previousOperandC, descriptor.memoryPrecisions!.C)
    
    // load  = directAccessCondition,
    // store = false
    //  problemSize = 1488 | A   B   |  832 threads/core | 8175 GFLOPS
    //  problemSize = 1488 | A   B^T | 1024 threads/core | 8712 GFLOPS
    //  problemSize = 1488 | A^T B   | 1024 threads/core | 8818 GFLOPS
    //  problemSize = 1488 | A^T B^T | 1024 threads/core | 8972 GFLOPS
    //  problemSize = 1489 | A   B   |  768 threads/core | 7888 GFLOPS
    //  problemSize = 1489 | A   B^T |  768 threads/core | 8256 GFLOPS
    //  problemSize = 1489 | A^T B   |  768 threads/core | 8026 GFLOPS
    //  problemSize = 1489 | A^T B^T |  832 threads/core | 8463 GFLOPS
    //
    // load  = directAccessCondition
    // store = directAccessCondition && (gid.y * M_group < M_edge) && (gid.x * N_group < N_edge)
    //  problemSize = 1488 | A   B   |  832 threads/core | 8186 GFLOPS
    //  problemSize = 1488 | A   B^T | 1024 threads/core | 8709 GFLOPS
    //  problemSize = 1488 | A^T B   | 1024 threads/core | 8808 GFLOPS
    //  problemSize = 1488 | A^T B^T | 1024 threads/core | 8984 GFLOPS
    //  problemSize = 1489 | A   B   |  768 threads/core | 7902 GFLOPS
    //  problemSize = 1489 | A   B^T |  768 threads/core | 8249 GFLOPS
    //  problemSize = 1489 | A^T B   |  768 threads/core | 8034 GFLOPS
    //  problemSize = 1489 | A^T B^T |  832 threads/core | 8469 GFLOPS
    //
    // load  = directAccessCondition && (gid.y * M_group < M_edge) && (gid.x * N_group < N_edge)
    // store = directAccessCondition && (gid.y * M_group < M_edge) && (gid.x * N_group < N_edge)
    //  problemSize = 1488 | A   B   |  832 threads/core | 8181 GFLOPS
    //  problemSize = 1488 | A   B^T | 1024 threads/core | 8710 GFLOPS
    //  problemSize = 1488 | A^T B   | 1024 threads/core | 8806 GFLOPS
    //  problemSize = 1488 | A^T B^T | 1024 threads/core | 8979 GFLOPS
    //  problemSize = 1489 | A   B   |  768 threads/core | 7892 GFLOPS
    //  problemSize = 1489 | A   B^T |  768 threads/core | 8242 GFLOPS
    //  problemSize = 1489 | A^T B   |  768 threads/core | 8034 GFLOPS
    //  problemSize = 1489 | A^T B^T |  832 threads/core | 8461 GFLOPS
    //
    // load previous C = false (M1 Max)
    //  problemSize = 1488 | A   B   |  896 threads/core | 8358 GFLOPS
    //  problemSize = 1488 | A   B^T | 1024 threads/core | 8682 GFLOPS
    //  problemSize = 1488 | A^T B   | 1024 threads/core | 8803 GFLOPS
    //  problemSize = 1488 | A^T B^T | 1024 threads/core | 9024 GFLOPS
    //  problemSize = 1489 | A   B   |  768 threads/core | 8039 GFLOPS
    //  problemSize = 1489 | A   B^T |  832 threads/core | 8376 GFLOPS
    //  problemSize = 1489 | A^T B   |  832 threads/core | 8374 GFLOPS
    //  problemSize = 1489 | A^T B^T |  832 threads/core | 8654 GFLOPS
    //
    // load previous C = true (M1 Max)
    //  problemSize = 1488 | A   B   |  896 threads/core | 8352 GFLOPS
    //  problemSize = 1488 | A   B^T |  896 threads/core | 8515 GFLOPS
    //  problemSize = 1488 | A^T B   | 1024 threads/core | 8760 GFLOPS
    //  problemSize = 1488 | A^T B^T | 1024 threads/core | 9007 GFLOPS
    //  problemSize = 1489 | A   B   |  768 threads/core | 7917 GFLOPS
    //  problemSize = 1489 | A   B^T |  768 threads/core | 7992 GFLOPS
    //  problemSize = 1489 | A^T B   |  832 threads/core | 8185 GFLOPS
    //  problemSize = 1489 | A^T B^T |  832 threads/core | 8583 GFLOPS
    //
    // load previous C = false (M4)
    //  problemSize = 1488 | A   B   | 1024 threads/core | 3353 GFLOPS
    //  problemSize = 1488 | A   B^T | 1024 threads/core | 3324 GFLOPS
    //  problemSize = 1488 | A^T B   | 1024 threads/core | 3338 GFLOPS
    //  problemSize = 1488 | A^T B^T | 1024 threads/core | 3289 GFLOPS
    //  problemSize = 1489 | A   B   | 1024 threads/core | 3375 GFLOPS
    //  problemSize = 1489 | A   B^T | 1024 threads/core | 3317 GFLOPS
    //  problemSize = 1489 | A^T B   | 1024 threads/core | 3343 GFLOPS
    //  problemSize = 1489 | A^T B^T | 1024 threads/core | 3298 GFLOPS
    //
    // load previous C = true (M4)
    //  problemSize = 1488 | A   B   | 1024 threads/core | 3374 GFLOPS
    //  problemSize = 1488 | A   B^T | 1024 threads/core | 3312 GFLOPS
    //  problemSize = 1488 | A^T B   | 1024 threads/core | 3321 GFLOPS
    //  problemSize = 1488 | A^T B^T | 1024 threads/core | 3249 GFLOPS
    //  problemSize = 1489 | A   B   | 1024 threads/core | 3323 GFLOPS
    //  problemSize = 1489 | A   B^T | 1024 threads/core | 3280 GFLOPS
    //  problemSize = 1489 | A^T B   | 1024 threads/core | 3308 GFLOPS
    //  problemSize = 1489 | A^T B^T | 1024 threads/core | 3256 GFLOPS
    
    // Profile the latency of matrix multiplication.
    for _ in 0..<(descriptor.loadPreviousC ? 1 : 15) {
      let duplicatedCommandCount: Int = (descriptor.loadPreviousC ? 1 : 20)
      
      // Encode the GPU command.
      let commandBuffer = MTLContext.global.commandQueue.makeCommandBuffer()!
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
  
  #if true
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
      var expected = leftSource - 2 * centerSource + rightSource
      if descriptor.loadPreviousC {
        if descriptor.transposeState!.A {
          expected += previousOperandC[n * problemSize + m]
        } else {
          expected += previousOperandC[m * problemSize + n]
        }
      }
      
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
  #endif
  return SIMD2(maxGFLOPS, occupancy)
}
#endif
