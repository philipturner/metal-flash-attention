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

#if true
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
    let matrixDimensions = (
      M: randomInts[0],
      N: randomInts[1],
      K: randomInts[2])
    let memoryPrecisions = (
      A: randomPrecision(),
      B: randomPrecision(),
      C: randomPrecision(),
      bias: randomPrecision())
    let transposeState = (
      A: Bool.random(),
      B: Bool.random(),
      bias: Bool.random())
    
    // Run a test.
    var gemmDesc = GEMMDescriptor()
    gemmDesc.matrixDimensions = matrixDimensions
    gemmDesc.memoryPrecisions = memoryPrecisions
    gemmDesc.transposeState = transposeState
    runCorrectnessTest(descriptor: gemmDesc)
  }
}

// Run a test with the specified configuration.
func runCorrectnessTest(descriptor: GEMMDescriptor) {
  guard let matrixDimensions = descriptor.matrixDimensions,
        let memoryPrecisions = descriptor.memoryPrecisions,
        let transposeState = descriptor.transposeState else {
    fatalError("Descriptor was incomplete.")
  }
  
  let checkpoint0 = CACurrentMediaTime()
  
  // Set the inputs.
  var operandA = [Float](
    repeating: .zero,
    count: Int(matrixDimensions.M * matrixDimensions.K))
  var operandB = [Float](
    repeating: .zero,
    count: Int(matrixDimensions.K * matrixDimensions.N))
  var operandBias = [Float](
    repeating: .zero,
    count: Int(transposeState.bias ? matrixDimensions.M : matrixDimensions.N))
  
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
  for elementID in operandBias.indices {
    let randomNumber = Float.random(in: 0..<1)
    operandBias[elementID] = randomNumber * normalizationFactor
  }
  
  // Create the buffers.
  var gpuOperandC = [Float](
    repeating: .zero,
    count: Int(matrixDimensions.M * matrixDimensions.N))
  let bufferA = MTLContext.global.createBuffer(operandA, memoryPrecisions.A)
  let bufferB = MTLContext.global.createBuffer(operandB, memoryPrecisions.B)
  let bufferC = MTLContext.global.createBuffer(gpuOperandC, memoryPrecisions.C)
  let bufferBias = MTLContext.global
    .createBuffer(operandBias, memoryPrecisions.bias)
  
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
    encoder.setBuffer(bufferBias, offset: 0, index: 3)
    
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
  do {
    let sourcePointer = bufferC.contents()
    for m in 0..<matrixDimensions.M {
      for n in 0..<matrixDimensions.N {
        let address = m * matrixDimensions.N + n
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
    count: Int(matrixDimensions.M * matrixDimensions.N))
  for m in 0..<matrixDimensions.M {
    for n in 0..<matrixDimensions.N {
      // Initialize with the bias value.
      var dotProduct: Float
      if transposeState.bias {
        dotProduct = operandBias[Int(m)]
      } else {
        dotProduct = operandBias[Int(n)]
      }
      
      // Execute the dot product.
      for k in 0..<matrixDimensions.K {
        var addressA: UInt32
        var addressB: UInt32
        if !transposeState.A {
          addressA = m * matrixDimensions.K + k
        } else {
          addressA = k * matrixDimensions.M + m
        }
        if !transposeState.B {
          addressB = k * matrixDimensions.N + n
        } else {
          addressB = n * matrixDimensions.K + k
        }
        
        let valueA = operandA[Int(addressA)]
        let valueB = operandB[Int(addressB)]
        dotProduct += valueA * valueB
      }
      
      // Store the result to memory.
      let addressC = m * matrixDimensions.N + n
      cpuOperandC[Int(addressC)] = dotProduct
    }
  }
  
  // Measure the Euclidean distance.
  var maxDistance: Float = .zero
  var totalDistance: Float = .zero
  for m in 0..<matrixDimensions.M {
    for n in 0..<matrixDimensions.N {
      let address = m * matrixDimensions.N + n
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
    C: GEMMOperandPrecision,
    bias: GEMMOperandPrecision),
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
  
  // registerPrecisions.C is FP16
  if memoryPrecisions.A == .FP16 &&
      memoryPrecisions.B == .FP16 &&
      memoryPrecisions.C == .FP16 {
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
  let biasPrecisions = [memoryPrecisions.C, memoryPrecisions.bias]
  if biasPrecisions.contains(.BF16) {
    tolerance += Float.exp2(-8.0)
  } else if biasPrecisions.contains(.FP16) {
    tolerance += Float.exp2(-10.0)
  } else {
    tolerance += Float.exp2(-23.0)
  }

  return tolerance
}
#endif

