//
//  AdversarialShapeTest.swift
//  FlashAttention
//
//  Created by Philip Turner on 6/21/24.
//

import Metal
import QuartzCore

// Test the correctness of the GEMM kernel, in edge cases where the matrix
// size is indivisible by the block size.

#if false
func executeScript() {
  print("Hello, console.")
  
  for _ in 0..<100 {
    var randomVecFloat = SIMD3<Float>.random(in: 0..<1)
    randomVecFloat = randomVecFloat * randomVecFloat * randomVecFloat
    var randomInts = SIMD3<UInt32>(randomVecFloat * 1000)
    randomInts.replace(with: .one, where: randomInts .== .zero)
    
    func randomPrecision() -> GEMMOperandPrecision {
      let randomInt = Int.random(in: 0..<3)
      switch randomInt {
      case 0:
        return .FP32
      case 1:
        return .FP16
      case 2:
        return .BF16
      default:
        fatalError("")
      }
    }
    
    // Define the problem configuration.
    let loadPreviousC = Bool.random()
    let matrixDimensions = (
      M: randomInts[0],
      N: randomInts[1],
      K: randomInts[2])
    let memoryPrecisions = (
      A: randomPrecision(),
      B: randomPrecision(),
      C: randomPrecision())
    let transposeState = (
      A: Bool.random(),
      B: Bool.random())
    
    // Set the leading dimensions.
    var leadingDimensions = (
      A: transposeState.A ? matrixDimensions.M : matrixDimensions.K,
      B: transposeState.B ? matrixDimensions.K : matrixDimensions.N,
      C: false ? matrixDimensions.M : matrixDimensions.N)
    if Bool.random() {
      leadingDimensions = (
        A: leadingDimensions.A + UInt32.random(in: 0..<64),
        B: leadingDimensions.B + UInt32.random(in: 0..<64),
        C: leadingDimensions.C + UInt32.random(in: 0..<64))
    }
    
    // Run a test.
    var gemmDesc = GEMMDescriptor()
    gemmDesc.leadingDimensions = leadingDimensions
    gemmDesc.loadPreviousC = loadPreviousC
    gemmDesc.matrixDimensions = matrixDimensions
    gemmDesc.memoryPrecisions = memoryPrecisions
    gemmDesc.transposeState = transposeState
    runCorrectnessTest(descriptor: gemmDesc)
  }
}

// Run a test with the specified configuration.
func runCorrectnessTest(descriptor: GEMMDescriptor) {
  guard let leadingDimensions = descriptor.leadingDimensions,
        let matrixDimensions = descriptor.matrixDimensions,
        let memoryPrecisions = descriptor.memoryPrecisions,
        let transposeState = descriptor.transposeState else {
    fatalError("Descriptor was incomplete.")
  }
  
  func chooseTrailingBlockDimension(
    _ transposeState: Bool,
    _ untransposedRows: UInt32,
    _ untransposedColumns: UInt32
  ) -> UInt32 {
    if transposeState {
      return untransposedColumns
    } else {
      return untransposedRows
    }
  }
  let trailingDimensionA = chooseTrailingBlockDimension(
    transposeState.A, matrixDimensions.M, matrixDimensions.K)
  let trailingDimensionB = chooseTrailingBlockDimension(
    transposeState.B, matrixDimensions.K, matrixDimensions.N)
  let trailingDimensionC = chooseTrailingBlockDimension(
    false, matrixDimensions.M, matrixDimensions.N)
  
  let checkpoint0 = CACurrentMediaTime()
  
  // Set the inputs.
  var operandA = [Float](
    repeating: .zero,
    count: Int(trailingDimensionA * leadingDimensions.A))
  var operandB = [Float](
    repeating: .zero,
    count: Int(trailingDimensionB * leadingDimensions.B))
  var operandPreviousC = [Float](
    repeating: .zero,
    count: Int(trailingDimensionC * leadingDimensions.C))
  
  // Normalize so that every dot product approaches 1.
  let normalizationFactor = 1 / Float(matrixDimensions.K).squareRoot()
  
  for elementID in operandA.indices {
    let randomNumber = Float.random(in: 0..<1)
    operandA[elementID] = randomNumber * normalizationFactor
  }
  for elementID in operandB.indices {
    let randomNumber = Float.random(in: 0..<1)
    operandB[elementID] = randomNumber * normalizationFactor
  }
  for elementID in operandPreviousC.indices {
    let randomNumber = Float.random(in: 0..<1)
    operandPreviousC[elementID] = randomNumber * normalizationFactor
  }
  
  // Create the buffers.
  let bufferA = MTLContext.global.createBuffer(operandA, memoryPrecisions.A)
  let bufferB = MTLContext.global.createBuffer(operandB, memoryPrecisions.B)
  let bufferC = MTLContext.global.createBuffer(
    operandPreviousC, memoryPrecisions.C)
  
  let checkpoint1 = CACurrentMediaTime()
  
  // Generate the kernel.
  let (kernel, pipeline) = GEMMKernel.fetchKernel(descriptor: descriptor)
  
  let checkpoint2 = CACurrentMediaTime()
  
  // Encode the GPU command.
  do {
    let commandQueue = MTLContext.global.commandQueue
    let commandBuffer = commandQueue.makeCommandBuffer()!
    let encoder = commandBuffer.makeComputeCommandEncoder()!
    encoder.setComputePipelineState(pipeline)
    encoder.setThreadgroupMemoryLength(
      Int(kernel.threadgroupMemoryAllocation), index: 0)
    encoder.setBuffer(bufferA, offset: 0, index: 0)
    encoder.setBuffer(bufferB, offset: 0, index: 1)
    encoder.setBuffer(bufferC, offset: 0, index: 2)
    
    func ceilDivide(_ target: UInt32, _ granularity: UInt16) -> Int {
      (Int(target) + Int(granularity) - 1) / Int(granularity)
    }
    let gridSize = MTLSize(
      width: ceilDivide(matrixDimensions.N, kernel.blockDimensions.N),
      height: ceilDivide(matrixDimensions.M, kernel.blockDimensions.M),
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
  
  // Copy the GPU output to an array.
  var gpuOperandC = [Float](
    repeating: .zero,
    count: Int(trailingDimensionC * leadingDimensions.C))
  do {
    let sourcePointer = bufferC.contents()
    for m in 0..<matrixDimensions.M {
      for n in 0..<matrixDimensions.N {
        let address = m * leadingDimensions.C + n
        var sourceValue: Float
        
        switch memoryPrecisions.C {
        case .FP32:
          let castedPointer = sourcePointer
            .assumingMemoryBound(to: Float.self)
          sourceValue = castedPointer[Int(address)]
        case .FP16:
          let castedPointer = sourcePointer
            .assumingMemoryBound(to: Float16.self)
          let sourceValue16 = castedPointer[Int(address)]
          sourceValue = Float(sourceValue16)
        case .BF16:
          let castedPointer = sourcePointer
            .assumingMemoryBound(to: UInt16.self)
          let sourceValue16 = castedPointer[Int(address)]
          let sourceValue16x2 = SIMD2<UInt16>(.zero, sourceValue16)
          sourceValue = unsafeBitCast(sourceValue16x2, to: Float.self)
        }
        gpuOperandC[Int(address)] = sourceValue
      }
    }
  }
  
  let checkpoint3 = CACurrentMediaTime()
  
  // Generate the output, on the reference implementation.
  var cpuOperandC = [Float](
    repeating: .zero,
    count: Int(trailingDimensionC * leadingDimensions.C))
  for m in 0..<matrixDimensions.M {
    for n in 0..<matrixDimensions.N {
      var dotProduct: Float = .zero
      
      for k in 0..<matrixDimensions.K {
        var addressA: UInt32
        var addressB: UInt32
        if !transposeState.A {
          addressA = m * leadingDimensions.A + k
        } else {
          addressA = k * leadingDimensions.A + m
        }
        if !transposeState.B {
          addressB = k * leadingDimensions.B + n
        } else {
          addressB = n * leadingDimensions.B + k
        }
        
        let valueA = operandA[Int(addressA)]
        let valueB = operandB[Int(addressB)]
        dotProduct += valueA * valueB
      }
      
      let addressC: UInt32 = m * leadingDimensions.C + n
      if descriptor.loadPreviousC {
        dotProduct += operandPreviousC[Int(addressC)]
      }
      cpuOperandC[Int(addressC)] = dotProduct
    }
  }
  
  // Measure the Euclidean distance.
  var maxDistance: Float = .zero
  var totalDistance: Float = .zero
  for m in 0..<matrixDimensions.M {
    for n in 0..<matrixDimensions.N {
      let address = m * leadingDimensions.C + n
      let cpuValue = cpuOperandC[Int(address)]
      let gpuValue = gpuOperandC[Int(address)]
      
      // Accumulate into the sum.
      let delta = gpuValue - cpuValue
      let distanceSquared = delta * delta
      let distance = distanceSquared.squareRoot()
      if distance > 1 {
        print(gpuValue, cpuValue)
      }
      maxDistance = max(maxDistance, distance)
      totalDistance += distance
    }
  }
  
  let checkpoint4 = CACurrentMediaTime()
  
  // Choose a tolerance for rounding error.
  let tolerance = createTolerance(
    memoryPrecisions: memoryPrecisions,
    accumulationDimension: matrixDimensions.K)
  
  let latencies = [
    checkpoint1 - checkpoint0,
    checkpoint2 - checkpoint1,
    checkpoint3 - checkpoint2,
    checkpoint4 - checkpoint3
  ]
  for latency in latencies {
    let repr = String(format: "%.1f", latency * 1000)
    print(repr, terminator: " ms | ")
  }
  print()
  
  guard maxDistance < tolerance else {
    fatalError("Failed correctness test for problem config: \(descriptor)\n\(maxDistance) \(tolerance)")
  }
}

fileprivate func createTolerance(
  memoryPrecisions: (
    A: GEMMOperandPrecision,
    B: GEMMOperandPrecision,
    C: GEMMOperandPrecision),
  accumulationDimension: UInt32
) -> Float {
  let precisions = [
    memoryPrecisions.A, memoryPrecisions.B, memoryPrecisions.C]
  let randomNoise = Float(accumulationDimension).squareRoot()
  
  // FP32 tolerance.
  var tolerance: Float = 3e-7
  
  // FP16 tolerance.
  if memoryPrecisions.A == .FP16 ||
      memoryPrecisions.B == .FP16 {
    tolerance = max(tolerance, 1e-5)
    tolerance = max(tolerance, 1e-3 / randomNoise)
  }
  if memoryPrecisions.C == .FP16 {
    tolerance = max(tolerance, 3e-4)
  }
  if precisions.allSatisfy({ $0 == .FP16 }) {
    tolerance = max(tolerance, 3e-3)
    tolerance = max(tolerance, 1e-5 * Float(accumulationDimension))
  }
  
  // BF16 tolerance.
  if precisions.contains(where: { $0 == .BF16 }) {
    if accumulationDimension < 1000 {
      tolerance = max(tolerance, 2e-2)
    } else {
      tolerance = max(tolerance, 5e-3)
    }
  }
  
  // List the precisions involved in the bias rounding event.
  // convert bias -> accumulator
  let biasPrecisions = [memoryPrecisions.C]
  if biasPrecisions.contains(.BF16) {
    tolerance += Float.exp2(-8.0)
  } else if biasPrecisions.contains(.FP16) {
    tolerance += Float.exp2(-10.0)
  } else {
    tolerance += Float.exp2(-22.0)
  }
  
  return tolerance
}
#endif

