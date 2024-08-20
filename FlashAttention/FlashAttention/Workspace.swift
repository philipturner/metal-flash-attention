//
//  Workspace.swift
//  FlashAttention
//
//  Created by Philip Turner on 6/20/24.
//

import Metal
import QuartzCore

/// The repo author's own workspace for running tests and developing kernels.
/// The contents of this function have no meaning, and ideally will be blank
/// when the 'main' branch is in a stable state. Clients can utilize this
/// function to script tests in their fork.

// Workspace for drafting the auto-parsing code.
func executeScript() {
  struct Row {
    var maximumHeadDimension: UInt16 = .zero
    var blockDimensionParallelization: UInt16 = .zero
    var blockDimensionTraversal: UInt16 = .zero
    var blockDimensionHead: UInt16 = .zero
    var cachedOperands: [AttentionOperand] = []
  }
  
  // Parse the table into rows.
  let rawString = """
  | 16  | 32 | 128 | 16 | Q, O |
  | 32  | 32 | 128 | 32 | Q, O |
  | 64  | 32 | 128 | 32 | Q, O |
  | 96  | 32 | 128 | 32 | O    |
  | 128 | 32 | 128 | 32 | Q    |
  | 160 | 32 | 128 | 32 | Q    |
  | 256 | 32 | 128 | 32 |      |
  | 384 | 32 | 128 | 32 |      |
  
  """
  
  // Split the lines by the newline delimiter.
  let lines = rawString.split(separator: "\n").map(String.init)
  
  // Iterate over the lines.
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
      fatalError("Number of segments was invalid.")
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
    
    rows.append(row)
  }
  
  // Display the rows.
  for row in rows {
    print(row.maximumHeadDimension, terminator: " | ")
    print(row.blockDimensionParallelization, terminator: " | ")
    print(row.blockDimensionTraversal, terminator: " | ")
    print(row.blockDimensionHead, terminator: " | ")
    
    for operand in row.cachedOperands {
      print(operand, terminator: " ")
    }
    print()
  }
}
