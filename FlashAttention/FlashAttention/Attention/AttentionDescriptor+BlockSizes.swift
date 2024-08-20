//
//  AttentionDescriptor+BlockSizes.swift
//  FlashAttention
//
//  Created by Philip Turner on 8/19/24.
//

import Metal

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
      | 16  | 32 | 64 | 16 | Q, O |
      | 32  | 32 | 64 | 32 | O    |
      | 64  | 32 | 64 | 64 | O    |
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
      | 32  | 32 | 32 | 32 | dQ    |
      | 64  | 32 | 64 | 16 | dQ    |
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

// MARK: - Parsing

struct Row {
  var maximumHeadDimension: UInt16 = .zero
  var blockDimensionParallelization: UInt16 = .zero
  var blockDimensionTraversal: UInt16 = .zero
  var blockDimensionHead: UInt16 = .zero
  var cachedOperands: [AttentionOperand] = []
}

/*private*/ func parseRows(_ rawString: String) -> [Row] {
  // Split the lines by the newline delimiter.
  let lines = rawString.split(separator: "\n").map(String.init)
  
  // Iterate over the lines.
  var output: [Row] = []
  for line in lines {
    // Split the cells by the horizontal bar delimiter.
    var segments = line.split(separator: "|").map(String.init)
    
    // Iterate over the cells.
    for segmentID in segments.indices {
      // Extract the null terminated C-style string.
      let segment = segments[segmentID]
      var characters = segment.utf8CString
      
      // Remove characters matching to the UTF-8 code for space.
      characters.removeAll(where: {
        $0 == 0x20
      })
      
      // Overwrite the segment with the trimmed one.
      let trimmedSegment = String(validatingUTF8: Array(characters))
      guard let trimmedSegment else {
        fatalError("UTF-8 was invalid.")
      }
      segments[segmentID] = trimmedSegment
    }
    
    // Check that the correct number of cells exist.
    guard segments.count == 5 else {
      fatalError("Number of segments was invalid: \(segments.count)")
    }
    
    // Utility function for parsing the integers.
    func parseInteger(_ string: String) -> UInt16 {
      let output = UInt16(string)
      guard let output else {
        fatalError("Invalid integer: \(string)")
      }
      return output
    }
    
    // Utility function for parsing the operands.
    func parseOperands(_ string: String) -> [AttentionOperand] {
      // Split the operands by the comma delimiter.
      let operandNames = string.split(separator: ",").map(String.init)
      
      // Using an O(keys * queries) algorithm to search through the accepted
      // operands. This likely has less overhead than creating a dictionary.
      // In addition, it is simpler to code.
      let acceptedOperands: [AttentionOperand] = [
        .Q, .K, .V, .O,
        .dO, .dV, .dK, .dQ
      ]
      
      // Iterate over the operand names.
      var output: [AttentionOperand] = []
      for operandName in operandNames {
        var matchedOperand: AttentionOperand?
        for operand in acceptedOperands {
          guard operand.description == operandName else {
            continue
          }
          matchedOperand = operand
        }
        
        guard let matchedOperand else {
          fatalError("Could not find match for \(operandName).")
        }
        output.append(matchedOperand)
      }
      
      return output
    }
    
    // Create a row object.
    var row = Row()
    row.maximumHeadDimension = parseInteger(segments[0])
    row.blockDimensionParallelization = parseInteger(segments[1])
    row.blockDimensionTraversal = parseInteger(segments[2])
    row.blockDimensionHead = parseInteger(segments[3])
    row.cachedOperands = parseOperands(segments[4])
    
    output.append(row)
  }
  
  return output
}
