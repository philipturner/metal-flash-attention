//
//  GEMMDescriptor.swift
//  FlashAttention
//
//  Created by Philip Turner on 6/21/24.
//

import func Metal.MTLCreateSystemDefaultDevice
import protocol Metal.MTLDevice

/// A description of a dense matrix-matrix multiplication.
struct GEMMDescriptor {
  /// The number of equally sized multiplications that run in parallel.
  /// Batching is out of scope for the reference implementation. However, there
  /// should be a guide for clients that wish to modify the shader, in ways
  /// that increase the compute workload. For example, by batching the
  /// multiplication of (sub)matrices located at arbitrary pointers in memory
  /// (with potentially nonuniform stride or noncontiguous padding).
  var batchDimension: Int = 1
  
  /// The dimensions of the input and output matrices.
  /// - Parameter M: Number of output columns.
  /// - Parameter N: Number of output rows.
  /// - Parameter K: Number of loop iterations for the dot products.
  ///
  /// For all practical purposes, one can assume matrix dimensions are 32-bit.
  /// I use this quite often in other code. The pointers themselves are 64-bit,
  /// but the offsets between different elements are 32-bit. With 4-byte words,
  /// this scheme could access up to 16 GB of memory - larger than any array
  /// in any reasonable application. Handling larger allocations likely
  /// requires consideration of more failure points than just integer
  /// overflows.
  var matrixDimensions: (M: UInt32, N: UInt32, K: UInt32)?
  
  var memoryPrecisions: (
    A: GEMMOperandPrecision, B: GEMMOperandPrecision, C: GEMMOperandPrecision)?
  
  var transposeState: (A: Bool, B: Bool)?
}

struct GEMMKey: Equatable, Hashable {
  var batchDimension: Int
  var matrixDimensions: SIMD3<UInt32>
  var memoryPrecisions: SIMD3<UInt16>
  var transposeState: SIMD2<UInt8>
 
  init(copying source: GEMMDescriptor) {
    batchDimension = source.batchDimension
    matrixDimensions = Self.createMatrixDimensions(source.matrixDimensions)
    memoryPrecisions = GEMMKernelKey.createPrecisions(source.memoryPrecisions)
    transposeState = GEMMKernelKey.createTransposeState(source.transposeState)
  }
  
  @_transparent // performance in -Ounchecked
  static func createMatrixDimensions(
    _ input: (UInt32, UInt32, UInt32)?
  ) -> SIMD3<UInt32> {
    if let input {
      return SIMD3(input.0, input.1, input.2)
    } else {
      return SIMD3(repeating: .max)
    }
  }
}

extension GEMMDescriptor: Hashable, Equatable {
  static func == (lhs: GEMMDescriptor, rhs: GEMMDescriptor) -> Bool {
    let lhsKey = GEMMKey(copying: lhs)
    let rhsKey = GEMMKey(copying: rhs)
    return lhsKey == rhsKey
  }
  
  func hash(into hasher: inout Hasher) {
    let key = GEMMKey(copying: self)
    hasher.combine(key)
  }
}

extension GEMMKernelDescriptor {
  /// Initialize the kernel descriptor using another descriptor, which just
  /// specifies the problem size. Then, forget the information about problem
  /// size. It will not be needed until the very far future, when the user
  /// retrieves a `MTLLibrary` from the cache and sets some Metal function
  /// constants.
  ///
  /// One might initialize a `GEMMKernelDescriptor` this way whenever an
  /// arbitrary matrix multiplication is requested. The generated descriptor
  /// itself could be a key in the KV cache. With this shader cache design, you
  /// must minimize the latency of actions like `MTLDevice` materialization and
  /// core count queries.
  ///
  /// Acceptable latency: no more than 1 μs per invocation.
  init(descriptor: GEMMDescriptor) {
    guard let matrixDimensions = descriptor.matrixDimensions,
          let memoryPrecisions = descriptor.memoryPrecisions,
          let transposeState = descriptor.transposeState else {
      fatalError("Descriptor was incomplete.")
    }
    
    // Select the only GPU on an Apple silicon system.
    //
    // NOTE: To avoid potentially costly API calls, you may wish to cache the
    // MTLDevice object or enter a previously created one. The core count
    // could also be cached on macOS.
    //
    // Typical latency to initiate a Metal device, provided the function has
    // been called numerous times prior:
    // - macOS 14
    //   - Swift debug mode,   Metal API validation on:  ≥33 μs
    //   - Swift release mode, Metal API validation off: ≥38 μs
    // - iOS 17
    //   - Swift debug mode,   Metal API validation on:   ≥0 μs
    //   - Swift release mode, Metal API validation off:  ≥0 μs
    let mtlDevice = MTLCreateSystemDefaultDevice()!
    
    // Trim the device name to something easier to process.
    //
    // M1 Max: Apple M1 Max -> M1
    // M4:     Apple M4 GPU -> M4
    func createDeviceName() -> String {
      let deviceName = mtlDevice.name
      var splits = deviceName.split(separator: " ").map(String.init)
      splits.removeAll(where: { $0.starts(with: "Apple") })
      splits.removeAll(where: { $0.starts(with: "GPU") })
      
      // Iterate over the space-separated words.
      var matchingSplitIDs: [UInt32] = []
      for splitID in splits.indices {
        // Screen out obvious non-candidates.
        let split = splits[splitID]
        guard split.starts(with: "A") || split.starts(with: "M") else {
          continue
        }
        
        // Extract the second character.
        guard split.count > 1 else {
          continue
        }
        let secondCharacterInt8 = split.utf8CString[1]
        let secondCharacterUInt32 = UInt32(secondCharacterInt8)
        let secondCharacterUnicode = Unicode.Scalar(secondCharacterUInt32)!
        let secondCharacter = Character(secondCharacterUnicode)
        
        // If the second character is numeric, the candidate passes.
        if secondCharacter.isWholeNumber {
          matchingSplitIDs.append(UInt32(splitID))
        }
      }
      guard matchingSplitIDs.count == 1 else {
        fatalError("Failed to locate device name.")
      }
      
      let splitID = matchingSplitIDs[0]
      return splits[Int(splitID)]
    }
    let deviceName = createDeviceName()
    
    // Find the core count.
#if os(macOS)
    // Typical latency to query IORegistry, provided the function has been
    // called numerous times prior:
    // - macOS 14
    //   - Swift debug mode,   Metal API validation on:  ≥9 μs
    //   - Swift release mode, Metal API validation off: ≥9 μs
    let coreCount = findCoreCount()
#elseif os(iOS)
    var coreCount: Int
    if deviceName.starts(with: "A") {
      if mtlDevice.supportsFamily(.apple9) {
        coreCount = 6
      } else {
        coreCount = 5
      }
    } else {
      coreCount = 10
    }
#endif
    
    // Select the register precisions.
    var registerPrecisionA = memoryPrecisions.A
    var registerPrecisionB = memoryPrecisions.B
    var registerPrecisionC = GEMMOperandPrecision.FP32
    if memoryPrecisions.A == .FP16,
       memoryPrecisions.B == .FP16,
       memoryPrecisions.C == .FP16 {
      registerPrecisionC = GEMMOperandPrecision.FP16
    }
    if !mtlDevice.supportsFamily(.apple9) {
      if memoryPrecisions.A == .BF16 {
        registerPrecisionA = .FP32
      }
      if memoryPrecisions.B == .BF16 {
        registerPrecisionB = .FP32
      }
    }
    
    // Set the properties of the 'GEMMKernelDescriptor' object.
    self.memoryPrecisions = memoryPrecisions
    if mtlDevice.supportsFamily(.apple9) {
      self.preferAsyncLoad = false
    } else {
      self.preferAsyncLoad = true
    }
    self.registerPrecisions = (
      registerPrecisionA,
      registerPrecisionB,
      registerPrecisionC)
    if !mtlDevice.supportsFamily(.apple9) {
      self.splits = (2, 2)
    } else {
      self.splits = (1, 1)
    }
    self.transposeState = transposeState
    
    // Set the properties that deal with block size.
    setBlockDimensions(
      mtlDevice: mtlDevice,
      coreCount: coreCount,
      matrixDimensions: matrixDimensions,
      batchDimension: descriptor.batchDimension)
  }
  
  // Implementation of the block size selection heuristic.
  //
  // This function initializes the 'blockDimensions' and
  // 'paddedBlockDimensions' properties.
  private mutating func setBlockDimensions(
    mtlDevice: MTLDevice,
    coreCount: Int,
    matrixDimensions: (M: UInt32, N: UInt32, K: UInt32),
    batchDimension: Int
  ) {
    guard let memoryPrecisions,
          let transposeState else {
      fatalError("Some properties were not set.")
    }
    guard !mtlDevice.supportsFamily(.apple9) else {
      self.blockDimensions = (32, 32, 8)
      return
    }
    
    // Find the actual number of threadgroups, with a large block size.
    func ceilDivide(_ target: UInt32, _ granularity: UInt16) -> UInt32 {
      (target + UInt32(granularity) - 1) / UInt32(granularity)
    }
    var actualGroups: Int = 1
    actualGroups *= Int(ceilDivide(matrixDimensions.M, 48))
    actualGroups *= Int(ceilDivide(matrixDimensions.N, 48))
    actualGroups *= Int(batchDimension)
    
    // Does the kernel use 48x48x24xFP32 (9 KB) or 48x48x32xFP16/BF16 (6 KB)?
    func requiresLargeAllocation(_ precision: GEMMOperandPrecision) -> Bool {
      switch precision {
      case .FP32: return true
      case .FP16: return false
      case .BF16: return false
      }
    }
    var useLargeAllocation = false
    if requiresLargeAllocation(memoryPrecisions.A) {
      useLargeAllocation = true
    }
    if requiresLargeAllocation(memoryPrecisions.B) {
      useLargeAllocation = true
    }
    if requiresLargeAllocation(memoryPrecisions.C) {
      useLargeAllocation = true
    }
    
    // Branch on whether the allocation is large / target occupancy is low.
    if useLargeAllocation {
      let idealGroups = coreCount * 6
      if actualGroups <= idealGroups {
        self.blockDimensions = (32, 32, 32)
      } else {
        self.blockDimensions = (48, 48, 24)
        
        // This is verified to be optimal for:
        // - (memA, memB, memC) = (FP32, FP32, FP32)
        // - (memA, memB, memC) = (FP16, FP16, FP32)
        // - (memA, memB, memC) = (FP16, FP32, FP32)
        // - (memA, memB, memC) = (FP16, FP32, FP16)
        switch transposeState {
        case (false, false):
          self.paddedBlockDimensions = ((48, 24), (24, 48), (48, 48))
        case (false, true):
          let paddedBK = (memoryPrecisions.B == .FP32) ? UInt16(28) : 24
          self.paddedBlockDimensions = ((48, 24), (paddedBK, 48), (48, 48))
        case (true, false):
          let paddedAM = (memoryPrecisions.A == .FP32) ? UInt16(52) : 56
          self.paddedBlockDimensions = ((paddedAM, 24), (24, 48), (48, 48))
        case (true, true):
          let paddedAM = (memoryPrecisions.A == .FP32) ? UInt16(52) : 56
          self.paddedBlockDimensions = ((paddedAM, 24), (24, 48), (48, 48))
        }
      }
    } else {
      let idealGroups = coreCount * 9
      if actualGroups <= idealGroups {
        blockDimensions = (32, 32, 32)
      } else {
        blockDimensions = (48, 48, 32)
      }
    }
  }
}
