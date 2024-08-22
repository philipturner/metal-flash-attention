//
//  AttentionParameterRow.swift
//  FlashAttention
//
//  Created by Philip Turner on 8/20/24.
//

struct AttentionParameterRow {
  // Operating range in the head dimension.
  var maximumHeadDimension: UInt16 = .zero
  
  // Block dimensions.
  var parallelization = String()
  var traversal = String()
  var head = String()
  
  // Operands cached in registers.
  var cachedOperands = String()
}

extension AttentionParameterRow {
  static func parseTable(_ file: String) -> [AttentionParameterRow] {
    // Split the lines by the newline delimiter.
    let lines = file.split(separator: "\n").map(String.init)
    
    // Iterate over the lines.
    var output: [AttentionParameterRow] = []
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
      
      // Decode the operating range.
      let maximumHeadDimension = UInt16(segments[0])
      guard let maximumHeadDimension else {
        fatalError("Could not extract maximum head dimension.")
      }
      
      // Initialize a row object.
      var row = AttentionParameterRow()
      row.maximumHeadDimension = maximumHeadDimension
      row.parallelization = segments[1]
      row.traversal = segments[2]
      row.head = segments[3]
      row.cachedOperands = segments[4]
      
      output.append(row)
    }
    
    return output
  }
  
  static func parseOperands(_ string: String) -> [AttentionOperand] {
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
}
