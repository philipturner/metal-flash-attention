//
//  GEMMKernel.swift
//  FlashAttention
//
//  Created by Philip Turner on 6/21/24.
//

import protocol Metal.MTLLibrary

struct GEMMKernel {
  // Compiled source code.
  var source: String = ""
  var library: MTLLibrary!
  
  // Address spaces and data types.
  var memoryPrecisions: (
    A: GEMMOperandPrecision, B: GEMMOperandPrecision, C: GEMMOperandPrecision)
  var preferAsyncLoad: Bool
  var preferAsyncStore: Bool
  var registerPrecisions: (
    A: GEMMOperandPrecision, B: GEMMOperandPrecision, C: GEMMOperandPrecision)
  
  // Layout of the data in memory.
  var blockBytes: (A: UInt16, B: UInt16, C: UInt16)
  var blockDimensions: (M: UInt16, N: UInt16, K: UInt16)
  var leadingBlockDimensions: (A: UInt16, B: UInt16, C: UInt16)
  var paddedBlockDimensionsA: (M: UInt16, K: UInt16)
  var paddedBlockDimensionsB: (K: UInt16, N: UInt16)
  var paddedBlockDimensionsC: (M: UInt16, N: UInt16)
  var transposeState: (A: Bool, B: Bool)
  
  // Threadgroup sizes.
  var registerM: UInt16
  var registerN: UInt16
  var splits: (M: UInt16, N: UInt16)
  var threadgroupSize: UInt16
  
  init(descriptor: GEMMKernelDescriptor) {
    guard let blockDimensions = descriptor.blockDimensions,
          let device = descriptor.device,
          let memoryPrecisions = descriptor.memoryPrecisions,
          let preferAsyncStore = descriptor.preferAsyncStore,
          let registerPrecisions = descriptor.registerPrecisions,
          let splits = descriptor.splits,
          let transposeState = descriptor.transposeState else {
      fatalError("Descriptor was incomplete: \(descriptor)")
    }
    
    self.blockDimensions = blockDimensions
    self.memoryPrecisions = memoryPrecisions
    self.preferAsyncLoad = descriptor.preferAsyncLoad
    self.preferAsyncStore = preferAsyncStore
    self.registerPrecisions = registerPrecisions
    self.splits = splits
    self.threadgroupSize = 32 * splits.M * splits.N
    self.transposeState = transposeState
    
    // Validate the correctness of register precisions.
    func checkOperandPair(
      memory: GEMMOperandPrecision,
      register: GEMMOperandPrecision
    ) -> Bool {
      // Truth table:
      //
      // memory | register | valid |
      // ------ | -------- | ----- |
      // FP32   | FP32     | yes   |
      // FP32   | FP16     | no    |
      // FP32   | BF16     | no    |
      // FP16   | FP32     | yes   |
      // FP16   | FP16     | yes   |
      // FP16   | BF16     | no    |
      // BF16   | FP32     | yes   |
      // BF16   | FP16     | no    |
      // BF16   | BF16     | yes   |
      //
      // Optimized form of the logic:
      //
      // If the register precision matches the memory precision,
      //   return true
      // If the register precision equals FP32,
      //   return true
      // Otherwise,
      //   return false
      //
      // The logic statements will change if you introduce custom quantized
      // formats. The truth table will grow exponentially. You'll need to add
      // more restrictions on accepted pairs to overcome the combinatorial
      // explosion.
      if register == memory {
        return true
      } else if register == .FP32 {
        return true
      } else {
        return false
      }
    }
    
    guard checkOperandPair(
      memory: memoryPrecisions.A, register: registerPrecisions.A) else {
      fatalError("Operand A had an invalid register precision.")
    }
    guard checkOperandPair(
      memory: memoryPrecisions.B, register: registerPrecisions.B) else {
      fatalError("Operand B had an invalid register precision.")
    }
    guard checkOperandPair(
      memory: memoryPrecisions.C, register: registerPrecisions.C) else {
      fatalError("Operand C had an invalid register precision.")
    }
    if registerPrecisions.C == .BF16 {
      // BF16 has too few mantissa bits to be an accurate accumulator. In
      // addition, switching from FP32 accumulator to BF16 accumulator slows
      // down execution speed on both M1/M2 and M3+.
      fatalError("BF16 cannot be used as the register precision for C.")
    }
    
    // Declare the size of M and N within a register allocation.
    registerM = blockDimensions.M / splits.M
    registerN = blockDimensions.N / splits.N
    
    // Retrieve the "padded" block dimensions, otherwise compute analytically
    // from the true block dimensions.
    if let paddedBlockDimensions = descriptor.paddedBlockDimensions {
      paddedBlockDimensionsA = paddedBlockDimensions.A
      paddedBlockDimensionsB = paddedBlockDimensions.B
      paddedBlockDimensionsC = paddedBlockDimensions.C
    } else {
      paddedBlockDimensionsA = (blockDimensions.M, blockDimensions.K)
      paddedBlockDimensionsB = (blockDimensions.K, blockDimensions.N)
      paddedBlockDimensionsC = (blockDimensions.M, blockDimensions.N)
    }
    
    // Determine the block dimensions from the transpose state.
    leadingBlockDimensions = (.zero, .zero, .zero)
    if transposeState.A {
      leadingBlockDimensions.A = paddedBlockDimensionsA.M
    } else {
      leadingBlockDimensions.A = paddedBlockDimensionsA.K
    }
    if transposeState.B {
      leadingBlockDimensions.B = paddedBlockDimensionsB.K
    } else {
      leadingBlockDimensions.B = paddedBlockDimensionsB.N
    }
    leadingBlockDimensions.C = paddedBlockDimensionsC.N
    
    // Determine the threadgroup memory allocation.
    do {
      blockBytes = (
        A: paddedBlockDimensionsA.M * paddedBlockDimensionsA.K,
        B: paddedBlockDimensionsB.K * paddedBlockDimensionsB.N,
        C: paddedBlockDimensionsC.M * paddedBlockDimensionsC.N)
      blockBytes.A *= UInt16(memoryPrecisions.A.size)
      blockBytes.B *= UInt16(memoryPrecisions.B.size)
      blockBytes.C *= UInt16(memoryPrecisions.C.size)
    }
    
    // Compile the shader source.
    source = createSource()
    library = try! device.makeLibrary(source: source, options: nil)
  }
}

extension GEMMKernel {
  func memoryName(_ operand: String) -> String {
    switch operand {
    case "A": return memoryPrecisions.A.name
    case "B": return memoryPrecisions.B.name
    case "C": return memoryPrecisions.C.name
    default:
      fatalError("Unrecognized operand.")
    }
  }
  
  func registerName(_ operand: String) -> String {
    switch operand {
    case "A": return registerPrecisions.A.name
    case "B": return registerPrecisions.B.name
    case "C": return registerPrecisions.C.name
    default:
      fatalError("Unrecognized operand.")
    }
  }
  
  var threadgroupMemoryAllocation: UInt16 {
    max(blockBytes.A + blockBytes.B, blockBytes.C)
  }
  
  func transposed(_ operand: String) -> Bool {
    switch operand {
    case "A": return transposeState.A
    case "B": return transposeState.B
    case "C": return false
    default: fatalError("Unrecognized operand.")
    }
  }
  
  func leadingDimension(_ operand: String) -> String {
    switch operand {
    case "A": return transposed("A") ? "M" : "K"
    case "B": return transposed("B") ? "K" : "N"
    case "C": return transposed("C") ? "M" : "N"
    default: fatalError("Unrecognized operand.")
    }
  }
}
