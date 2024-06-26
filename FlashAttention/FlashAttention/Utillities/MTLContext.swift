//
//  MTLContext.swift
//  FlashAttention
//
//  Created by Philip Turner on 6/26/24.
//

import Metal

struct MTLContext {
  var device: MTLDevice
  var commandQueue: MTLCommandQueue
  
  static let global = MTLContext()
  
  init() {
    device = MTLCreateSystemDefaultDevice()!
    commandQueue = device.makeCommandQueue()!
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
}
