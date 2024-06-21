//
//  Workspace.swift
//  FlashAttention
//
//  Created by Philip Turner on 6/20/24.
//

import Metal

/// The repo author's own workspace for running tests and developing kernels.
/// The contents of this function have no meaning, and ideally will be blank
/// when the 'main' branch is in a stable state. Clients can utilize this
/// function to script tests in their fork.
func executeScript() {
  print("Hello, console.")
  
  var randomVecFloat = SIMD3<Float>.random(in: 0..<1)
  randomVecFloat = randomVecFloat * randomVecFloat * randomVecFloat
  var randomInts = SIMD3<UInt32>(randomVecFloat * 1000)
  randomInts.replace(with: .one, where: randomInts .== .zero)
  
  // Define the problem configuration.
  let matrixDimensions = (
    M: randomInts[0],
    N: randomInts[1],
    K: randomInts[2])
  let transposeState = (
    A: Bool.random(),
    B: Bool.random())
  
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
    _ originalData: [Float]
  ) -> MTLBuffer {
    // Add random numbers to expose out-of-bounds accesses.
    var augmentedData = originalData
    for _ in 0..<originalData.count {
      let randomNumber = Float.random(in: -2...2)
      augmentedData.append(randomNumber)
    }
    
    // Allocate enough memory to store everything in Float32.
    let bufferSize = augmentedData.count * 4
    let buffer = MTLContext.global.device.makeBuffer(length: bufferSize)!
    
    // Copy the data into the buffer.
    let pointer = buffer.contents().assumingMemoryBound(to: Float.self)
    for i in augmentedData.indices {
      pointer[i] = augmentedData[i]
    }
    return buffer
  }
  
  // Create the buffers.
  var gpuOperandC = [Float](
    repeating: .zero,
    count: Int(matrixDimensions.M * matrixDimensions.N))
  let bufferA = createBuffer(operandA)
  let bufferB = createBuffer(operandB)
  let bufferC = createBuffer(gpuOperandC)
  
  // Generate the kernel.
  var gemmDesc = GEMMDescriptor()
  gemmDesc.matrixDimensions = matrixDimensions
  gemmDesc.memoryPrecisions = (.FP32, .FP32, .FP32)
  gemmDesc.transposeState = transposeState
  let (kernel, pipeline) = retrieveGEMMKernel(descriptor: gemmDesc)
  
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
    let sourcePointer = bufferC.contents().assumingMemoryBound(to: Float.self)
    for m in 0..<matrixDimensions.M {
      for n in 0..<matrixDimensions.N {
        let address = m * matrixDimensions.N + n
        let sourceValue = sourcePointer[Int(address)]
        gpuOperandC[Int(address)] = sourceValue
      }
    }
  }
  
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
  for m in 0..<matrixDimensions.M {
    for n in 0..<matrixDimensions.N {
      let address = m * matrixDimensions.N + n
      let cpuValue = cpuOperandC[Int(address)]
      let gpuValue = gpuOperandC[Int(address)]
      
      // Accumulate into the sum.
      let delta = gpuValue - cpuValue
      let distanceSquared = delta * delta
      let distance = distanceSquared.squareRoot()
      maxDistance = max(maxDistance, distance)
    }
  }
  
  // Choose a threshold.
  let averageMagnitude = Float(0.25) * Float(matrixDimensions.K)
  let averageDeviation = Float(0.5) * Float(matrixDimensions.K).squareRoot()
  let tolerance = max(1e-5 * averageMagnitude, 6e-7 * averageDeviation)
  print(matrixDimensions)
  print(1e-5 * averageMagnitude, 6e-7 * averageDeviation)
  print(maxDistance, tolerance)
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
