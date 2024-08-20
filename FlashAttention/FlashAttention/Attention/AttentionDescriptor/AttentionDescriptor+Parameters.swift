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
    var createParameters: (MTLDevice) -> String
    
    // Branch on the type.
    switch type {
    case .forward:
      if lowPrecisionInputs && lowPrecisionIntermediates {
        createParameters = Self.forwardHalfPrecision(device:)
      } else {
        createParameters = Self.forwardSinglePrecision(device:)
      }
    case .backwardQuery:
      createParameters = Self.backwardQuery(device:)
    case .backwardKeyValue:
      createParameters = Self.backwardKeyValue(device:)
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
      | 0   | 32 | 64  | 32 |      |
      
      """
    }
  }
  
  /// Block sizes and cached operands for FP16 forward.
  ///
  /// Requires the following statements to be true. If any operand does not
  /// match the specified precision, use the heuristics for FP32 forward.
  ///
  /// ```swift
  /// registerPrecisions[.Q] = .FP16
  /// registerPrecisions[.K] = .FP16
  /// registerPrecisions[.V] = .FP16
  /// registerPrecisions[.S] = .FP16
  /// registerPrecisions[.P] = .FP16
  /// registerPrecisions[.O] = .FP32
  /// ```
  static func forwardHalfPrecision(device: MTLDevice) -> String {
    if device.supportsFamily(.apple9) {
      return """
      | 16  | 16 | 128 | 16 | Q, O |
      | 32  | 16 | 128 | 16 | Q, O |
      | 64  | 16 | 128 | 32 | Q, O |
      | 96  | 16 | 128 | 32 | Q, O |
      | 128 | 16 | 128 | 32 | O    |
      | 160 | 16 | 128 | 32 | O    |
      | 256 | 16 | 128 | 32 |      |
      
      """
    } else {
      return """
      | 16  | 32 | 128 | 16 | Q, O |
      | 32  | 32 | 128 | 32 | Q, O |
      | 64  | 32 | 128 | 32 | Q, O |
      | 96  | 32 | 128 | 32 | O    |
      | 128 | 32 | 128 | 32 | Q    |
      | 160 | 32 | 128 | 32 | Q    |
      | 256 | 32 | 128 | 32 |      |
      | 384 | 32 | 128 | 32 |      |
      
      """
    }
  }
  
  /// Block sizes and cached operands for FP32 forward.
  static func forwardSinglePrecision(device: MTLDevice) -> String {
    if device.supportsFamily(.apple9) {
      return """
      | 16  | 16 | 128 | 16 | Q, O |
      | 32  | 16 | 128 | 32 | Q, O |
      | 64  | 16 | 128 | 16 | Q, O |
      | 128 | 16 | 128 | 16 | O    |
      | 160 | 16 | 128 | 16 | O    |
      | 192 | 16 | 64  | 16 | O    |
      | 256 | 16 | 48  | 16 | O    |
      | 384 | 16 | 128 | 8  |      |
      
      """
    } else {
      return """
      | 24  | 32 | 64 | 24 | Q, O |
      | 32  | 32 | 64 | 32 | O    |
      | 56  | 32 | 32 | 64 | O    |
      | 128 | 32 | 64 | 32 |      |
      | 256 | 32 | 64 | 32 |      |
      
      """
    }
  }
  
  /// Block sizes and cached operands for FP32 backward query.
  static func backwardQuery(device: MTLDevice) -> String {
    if device.supportsFamily(.apple9) {
      return """
      | 16  | 16 | 64  | 32 | Q, dQ |
      | 32  | 16 | 64  | 32 | Q, dQ |
      | 64  | 16 | 64  | 32 | Q, dQ |
      | 128 | 16 | 64  | 32 | Q, dQ |
      | 256 | 16 | 64  | 32 | dQ    |
      | 384 | 16 | 128 | 8  |       |
      
      """
    } else {
      return """
      | 16  | 32 | 64 | 16 | Q, dQ |
      | 56  | 32 | 64 | 32 | dQ    |
      | 128 | 32 | 64 | 32 |       |
      | 256 | 32 | 64 | 32 |       |
      
      """
    }
  }
  
  /// Block sizes and cached operands for FP32 backward key-value.
  static func backwardKeyValue(device: MTLDevice) -> String {
    if device.supportsFamily(.apple9) {
      return """
      | 16  | 16 | 64  | 8  | K, V, dV, dK |
      | 32  | 16 | 32  | 16 | K, V, dV, dK |
      | 64  | 16 | 32  | 16 | V, dV, dK    |
      | 128 | 16 | 128 | 16 | dV, dK       |
      | 256 | 16 | 128 | 8  | dV           |
      | 384 | 16 | 128 | 8  |              |
      
      """
    } else {
      return """
      | 16  | 32 | 32 | 16 | V, dV, dK |
      | 32  | 32 | 64 | 16 | dV        |
      | 64  | 32 | 64 | 16 | dV        |
      | 256 | 32 | 64 | 16 |           |
      
      """
    }
  }
}
