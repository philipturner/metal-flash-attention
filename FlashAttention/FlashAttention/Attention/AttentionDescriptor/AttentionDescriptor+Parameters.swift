//
//  AttentionDescriptor+Parameters.swift
//  FlashAttention
//
//  Created by Philip Turner on 8/19/24.
//

import Metal

// MARK: - API

extension AttentionDescriptor {
  func parameterFile(type: AttentionKernelType) -> String {
    // Choose a function pointer for the parameters.
    var createParameters: (MTLDevice) -> String
    if lowPrecisionInputs && lowPrecisionIntermediates {
      switch type {
      case .forward: 
        createParameters = Self.forwardMixed(device:)
      case .backwardQuery:
        createParameters = Self.backwardQueryMixed(device:)
      case .backwardKeyValue:
        createParameters = Self.backwardKeyValueMixed(device:)
      }
    } else {
      switch type {
      case .forward: 
        createParameters = Self.forward(device:)
      case .backwardQuery:
        createParameters = Self.backwardQuery(device:)
      case .backwardKeyValue:
        createParameters = Self.backwardKeyValue(device:)
      }
    }
    
    // Retrieve the parameter file.
    let device = MTLContext.global.device
    return createParameters(device)
  }
  
  func row(table: [AttentionParameterRow]) -> AttentionParameterRow {
    guard let matrixDimensions = self.matrixDimensions else {
      fatalError("Descriptor was incomplete.")
    }
    let headDimension = matrixDimensions.D
    
    // Pick a row of the table.
    //
    // Searching for a head dimension that exceeds the specified one.
    var matchedRowID: Int?
    for rowID in table.indices {
      let row = table[rowID]
      if headDimension <= row.maximumHeadDimension {
        // Quit on the first match.
        matchedRowID = rowID
        break
      }
    }
    
    // Extract the row from the table.
    if let matchedRowID {
      return table[matchedRowID]
    } else {
      return table.last!
    }
  }
}

// MARK: - Parameters

extension AttentionDescriptor {
  /// Default parameters, which should yield reasonable performance in
  /// any context.
  ///
  /// Use this as a fallback option, in contexts where the fine-tuned
  /// parameters do not generalize.
  static func defaultParameters(device: MTLDevice) -> String {
    if device.supportsFamily(.apple9) {
      return """
      | 0   | 16 | 128 | 16 |      |
      
      """
    } else {
      return """
      | 0   | 32 | 80  | 16 |      |
      
      """
    }
  }
  
  /// Block sizes and cached operands for FP16 forward.
  ///
  /// Requires the following statements to be true. If any operand does not
  /// match the specified precision, the parameters will fail to generalize.
  ///
  /// ```swift
  /// registerPrecisions[.Q] = .FP16
  /// registerPrecisions[.K] = .FP16
  /// registerPrecisions[.S] = .FP16
  /// registerPrecisions[.P] = .FP16
  /// registerPrecisions[.V] = .FP16
  /// registerPrecisions[.O] = .FP32
  ///
  /// memoryPrecisions[.L] = .FP16
  /// ```
  static func forwardMixed(device: MTLDevice) -> String {
    if device.supportsFamily(.apple9) {
      return """
      | 32  | 16 | 128 | 16 | Q, O |
      | 96  | 16 | 128 | 32 | Q, O |
      | 160 | 16 | 128 | 32 | O    |
      | 224 | 16 | 128 | 32 | Q    |
      | 384 | 16 | 128 | 32 |      |
      
      """
    } else {
      return """
      | 96  | 32 | 128 | 32 | Q, O |
      | 128 | 32 | 128 | 32 | Q    |
      | 384 | 32 | 128 | 32 |      |
      
      """
    }
  }
  
  /// Block sizes and cached operands for FP32 forward.
  ///
  /// If any of the operands is FP16, the parameters will fail to generalize.
  static func forward(device: MTLDevice) -> String {
    if device.supportsFamily(.apple9) {
      return """
      | 8   | 16 | 128 | 16 | Q, O |
      | 16  | 16 | 64  | 16 | Q, O |
      | 48  | 16 | 32  | 8  | Q, O |
      | 192 | 16 | 64  | 16 | O    |
      | 384 | 16 | 128 | 16 |      |
      
      """
    } else {
      return """
      | 24  | 32 | 64 | 24 | Q, O |
      | 32  | 32 | 64 | 32 | O    |
      | 56  | 32 | 32 | 56 | O    |
      | 384 | 32 | 80 | 16 |      |
      
      """
    }
  }
}

extension AttentionDescriptor {
  /// Block sizes and cached operands for FP16/BF16 backward query.
  ///
  /// Requires the following statements to be true. If any operand does not
  /// match the specified precision, the parameters will fail to generalize.
  ///
  /// ```swift
  /// registerPrecisions[.Q] = .FP16
  /// registerPrecisions[.K] = .FP16
  /// registerPrecisions[.S] = .FP16
  /// registerPrecisions[.P] = .FP16
  /// registerPrecisions[.V] = .FP16
  /// registerPrecisions[.O] = .FP32
  ///
  /// memoryPrecisions[.D] = .BF16
  ///
  /// registerPrecisions[.dO] = .BF16 (M3) .FP32 (M1)
  /// registerPrecisions[.dP] = .FP32
  /// registerPrecisions[.dS] = .BF16 (M3) .FP32 (M1)
  /// registerPrecisions[.dQ] = .FP32
  /// ```
  static func backwardQueryMixed(device: MTLDevice) -> String {
    if device.supportsFamily(.apple9) {
      return """
      | 80  | 16 | 64  | 8  | Q, dO, dQ |
      | 192 | 16 | 64  | 32 | Q, dQ     |
      | 384 | 16 | 128 | 32 |           |
      
      """
    } else {
      return """
      | 32  | 32 | 64 | 32 | Q, dQ |
      | 96  | 32 | 64 | 32 | dQ    |
      | 384 | 32 | 64 | 32 |       |
      
      """
    }
  }
  
  /// Block sizes and cached operands for FP32 backward query.
  ///
  /// If any of the operands is FP16 or BF16, the parameters will fail to
  /// generalize.
  static func backwardQuery(device: MTLDevice) -> String {
    if device.supportsFamily(.apple9) {
      return """
      | 16  | 16 | 64  | 8  | Q, dO, dQ |
      | 32  | 16 | 64  | 16 | Q, dQ     |
      | 192 | 16 | 64  | 32 | Q, dQ     |
      | 384 | 16 | 128 | 16 |           |
      
      """
    } else {
      return """
      | 16  | 32 | 64 | 16 | Q, dQ |
      | 32  | 32 | 64 | 32 | dQ    |
      | 56  | 32 | 64 | 24 | dQ    |
      | 384 | 32 | 80 | 16 |       |
      
      """
    }
  }
}

extension AttentionDescriptor {
  /// Block sizes and cached operands for FP16/BF16 backward query.
  ///
  /// Requires the following statements to be true. If any operand does not
  /// match the specified precision, the parameters will fail to generalize.
  ///
  /// ```swift
  /// registerPrecisions[.Q] = .FP16
  /// registerPrecisions[.K] = .FP16
  /// registerPrecisions[.S] = .FP16
  /// registerPrecisions[.P] = .FP16
  /// registerPrecisions[.V] = .FP16
  /// registerPrecisions[.O] = .FP32
  ///
  /// registerPrecisions[.L] = .FP16
  /// registerPrecisions[.D] = .BF16 (M3) .FP32 (M1)
  ///
  /// registerPrecisions[.dO] = .BF16 (M3) .FP32 (M1)
  /// registerPrecisions[.dV] = .FP32
  /// registerPrecisions[.dP] = .FP32
  /// registerPrecisions[.dS] = .BF16 (M3) .FP32 (M1)
  /// registerPrecisions[.dK] = .FP32
  /// registerPrecisions[.dQ] = .FP32
  /// ```
  static func backwardKeyValueMixed(device: MTLDevice) -> String {
    if device.supportsFamily(.apple9) {
      return """
      | 56  | 16 | 64  | 8  | K, V, dV, dK |
      | 80  | 16 | 32  | 16 | V, dV, dK    |
      | 144 | 16 | 128 | 16 | dV, dK       |
      | 224 | 16 | 128 | 16 | dV           |
      | 384 | 16 | 128 | 32 |              |
      
      """
    } else {
      return """
      | 16  | 32 | 64 | 16 | V, dV, dK |
      | 32  | 32 | 64 | 32 | dV, dK    |
      | 96  | 32 | 64 | 32 | dV        |
      | 384 | 32 | 64 | 32 |           |
      
      """
    }
  }
  
  /// Block sizes and cached operands for FP32 backward key-value.
  ///
  /// If any of the operands is FP16 or BF16, the parameters will fail to
  /// generalize.
  static func backwardKeyValue(device: MTLDevice) -> String {
    if device.supportsFamily(.apple9) {
      return """
      | 16  | 16 | 64  | 8  | K, V, dV, dK |
      | 32  | 16 | 32  | 16 | K, V, dV, dK |
      | 64  | 16 | 32  | 16 | V, dV, dK    |
      | 128 | 16 | 128 | 16 | dV, dK       |
      | 160 | 16 | 128 | 16 | dV           |
      | 384 | 16 | 128 | 16 |              |
      
      """
    } else {
      return """
      | 16  | 32 | 32 | 16 | V, dV, dK |
      | 24  | 32 | 64 | 24 | dV, dK    |
      | 56  | 32 | 80 | 16 | dV        |
      | 384 | 32 | 80 | 16 |           |
      
      """
      
      /*
       8192, 72x16
       36, 2732, 2081, 1787
       40, 4047, 3341, 2792
       44, 3017, 2094, 2080
       48, 4158, 3506, 3325
       52, 2934, 2083, 2124
       56, 4083, 3342, 2891
       
       8192, 80x16
       36, 2732, 2079, 1931
       40, 4042, 3341, 2871
       44, 2981, 2098, 2096
       48, 4159, 3504, 3208
       52, 2938, 2084, 2110
       56, 4085, 3339, 3047
       
       8192, 72x16, TTTT
       36, 3441, 3335, 2576
       40, 3840, 3581, 3052
       44, 3592, 2932, 2850
       48, 3847, 3549, 3470
       52, 3324, 2906, 2645
       56, 3696, 3287, 3085
       
       8192, 80x16, TTTT
       36, 3447, 3336, 2601
       40, 3843, 3584, 3158
       44, 3591, 2933, 2930
       48, 3848, 3551, 3373
       52, 3319, 2909, 2720
       56, 3695, 3277, 3298
       
       16384, 72x16
       40, 4038, 2898, 2558
       48, 4149, 3043, 3115
       56, 4067, 2907, 2801
       
       16384, 80x16
       40, 4037, 2902, 2597
       48, 4148, 2950, 3082
       56, 4071, 2794, 2823
       */
    }
  }
}
