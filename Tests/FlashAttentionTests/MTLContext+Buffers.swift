import FlashAttention
import Metal

extension MTLContext {
  func createBuffer(
    _ originalData: [Float],
    _ precision: GEMMOperandPrecision
  ) -> MTLBuffer {
    // Add random numbers to expose out-of-bounds accesses.
    var augmentedData = originalData
    
    // Avoid exceeding the maximum buffer allocation size.
    if originalData.count * 4 < 1_000_000_000 {
      for _ in 0..<originalData.count {
        let randomNumber = Float.random(in: -20...20)
        augmentedData.append(randomNumber)
      }
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
  
  static func copy(
    _ buffer: MTLBuffer,
    into array: inout [Float],
    precision: GEMMOperandPrecision = .FP32
  ) {
    let expectedLength = array.count * precision.size
    guard buffer.length >= expectedLength else {
      fatalError("Buffer was too small.")
    }
            
    let raw = buffer.contents()
    for elementID in array.indices {
      let address = elementID
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
      array[address] = entry32
    }
  }
}
