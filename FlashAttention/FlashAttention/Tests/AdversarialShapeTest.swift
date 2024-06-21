//
//  AdversarialShapeTest.swift
//  FlashAttention
//
//  Created by Philip Turner on 6/21/24.
//

import Metal
import QuartzCore

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
        return .FP32
      default:
        fatalError("")
      }
    }
    
    // Define the problem configuration.
    let matrixDimensions = (
      M: randomInts[0],
      N: randomInts[1],
      K: randomInts[2])
    let transposeState = (
      A: Bool.random(),
      B: Bool.random())
    let memoryPrecisions = (
      A: randomPrecision(),
      B: GEMMOperandPrecision.FP32,
      C: GEMMOperandPrecision.FP32)
    
    // Run a test.
    var gemmDesc = GEMMDescriptor()
    gemmDesc.matrixDimensions = (8, 8, 10_000)
    gemmDesc.memoryPrecisions = (.FP16, .FP16, .FP16)
    gemmDesc.transposeState = (false, false)
    runCorrectnessTest(descriptor: gemmDesc)
  }
}

struct MTLContext {
  var device: MTLDevice
  var commandQueue: MTLCommandQueue
  
  init() {
    device = MTLCreateSystemDefaultDevice()!
    commandQueue = device.makeCommandQueue()!
  }
  
  static let global = MTLContext()
}

// Run a test with the specified configuration.
func runCorrectnessTest(descriptor: GEMMDescriptor) {
  print()
  guard let matrixDimensions = descriptor.matrixDimensions,
        let memoryPrecisions = descriptor.memoryPrecisions,
        let transposeState = descriptor.transposeState else {
    fatalError("Descriptor was incomplete.")
  }
  
  let checkpoint0 = CACurrentMediaTime()
  
  // Set the outputs.
  var operandA = [Float](
    repeating: .zero,
    count: Int(matrixDimensions.M * matrixDimensions.K))
  var operandB = [Float](
    repeating: .zero,
    count: Int(matrixDimensions.K * matrixDimensions.N))
  for elementID in operandA.indices {
    let randomNumber = Float.random(in: 0..<1)
    operandA[elementID] = randomNumber
  }
  for elementID in operandB.indices {
    let randomNumber = Float.random(in: 0..<1)
    operandB[elementID] = randomNumber
  }
  
  func createBuffer(
    _ originalData: [Float],
    _ precision: GEMMOperandPrecision
  ) -> MTLBuffer {
    // Add random numbers to expose out-of-bounds accesses.
    var augmentedData = originalData
    for _ in 0..<originalData.count {
      let randomNumber = Float.random(in: -20...20)
      augmentedData.append(randomNumber)
    }
    
    // Allocate enough memory to store everything in Float32.
    let bufferSize = augmentedData.count * 4
    let device = MTLContext.global.device
    let buffer = device.makeBuffer(length: bufferSize)!
    
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
  
  // Create the buffers.
  var gpuOperandC = [Float](
    repeating: .zero,
    count: Int(matrixDimensions.M * matrixDimensions.N))
  let bufferA = createBuffer(operandA, memoryPrecisions.A)
  let bufferB = createBuffer(operandB, memoryPrecisions.B)
  let bufferC = createBuffer(gpuOperandC, memoryPrecisions.C)
  
  let checkpoint1 = CACurrentMediaTime()
  
  // Generate the kernel.
  let (kernel, pipeline) = retrieveGEMMKernel(descriptor: descriptor)
  
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
      var dotProduct: Float = .zero
      
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
  print(maxDistance, tolerance)
  print(totalDistance, tolerance * Float(matrixDimensions.M * matrixDimensions.N))
  
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
    fatalError("Failed correctness test for problem config: \(descriptor)")
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
  
  // O(n) truncation error of the inputs
  // - Dominates for large K
  let averageMagnitude = Float(0.25) * Float(accumulationDimension)
  var magnitudeFactor: Float = 1e-6
  if memoryPrecisions.A == .FP16 ||
      memoryPrecisions.B == .FP16 {
    magnitudeFactor = max(magnitudeFactor, 1e-4)
  }
  if memoryPrecisions.C == .FP16 {
    magnitudeFactor = max(magnitudeFactor, 1e-3)
  }
  if precisions.allSatisfy({ $0 == .FP16 }) {
    magnitudeFactor = max(magnitudeFactor, 1e-1)
  }
  
  // O(sqrt(n)) accumulated rounding error
  // - Dominates for small K
  let averageDeviation = Float(0.5) * Float(accumulationDimension).squareRoot()
  var deviationFactor: Float = 0
  if memoryPrecisions.A == .FP16 ||
      memoryPrecisions.B == .FP16 {
    deviationFactor = max(deviationFactor, 1e-3)
  }
  if memoryPrecisions.C == .FP16 {
    deviationFactor = max(deviationFactor, 3e-3)
  }
  
  return max(
    magnitudeFactor * averageMagnitude,
    deviationFactor * averageDeviation)
}
#endif

