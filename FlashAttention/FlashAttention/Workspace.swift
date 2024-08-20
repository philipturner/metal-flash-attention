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
  let sequenceDimension: Int = 1024
  let headDimension: Int = 12
  
  var attentionDesc = AttentionDescriptor()
  attentionDesc.lowPrecisionInputs = true
  attentionDesc.lowPrecisionIntermediates = true
  attentionDesc.matrixDimensions = (
    R: UInt32(sequenceDimension),
    C: UInt32(sequenceDimension),
    D: UInt16(headDimension))
  attentionDesc.transposeState = (Q: false, K: false, V: false, O: false)
  
  // Fetch the parameters.
  let file = attentionDesc.parameterFile(type: .backwardKeyValue)
  let table = AttentionParameterRow.parseTable(file)
  let row = attentionDesc.row(table: table)
  
  let blockDimensions = row.createBlockDimensions()
  let operands = AttentionParameterRow.parseOperands(row.cachedOperands)
  
  // Display the selected parameters.
  print()
  print("maximum head dimension:", row.maximumHeadDimension)
  print("block dimension (R):", blockDimensions[0])
  print("block dimension (C):", blockDimensions[1])
  print("block dimension (D):", blockDimensions[2])
  print("cached operands:", operands)
  print()
}
