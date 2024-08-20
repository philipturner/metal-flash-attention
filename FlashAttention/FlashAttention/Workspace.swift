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
  // Specify the head dimension.
  let headDimension: UInt16 = 40
  
  // Load the parameters.
  let device = MTLCreateSystemDefaultDevice()!
  let parameters = AttentionDescriptor
    .backwardQuery(device: device)
  let rows = parseRows(parameters)
  
  // Pick a row of the table.
  var matchedRowID: Int?
  for rowID in rows.indices {
    let row = rows[rowID]
    if headDimension <= row.maximumHeadDimension {
      // Quit on the first match.
      matchedRowID = rowID
      break
    }
  }
  
  // Extract the row from the table.
  var row: Row
  if let matchedRowID {
    row = rows[matchedRowID]
  } else {
    row = rows.last!
  }
  
  // Display the selected parameters.
  print()
  print("maximum head dimension:", row.maximumHeadDimension)
  print("block dimension (R):", row.blockDimensionParallelization)
  print("block dimension (C):", row.blockDimensionTraversal)
  print("block dimension (D):", row.blockDimensionHead)
  print("cached operands:", row.cachedOperands)
  print()
}
