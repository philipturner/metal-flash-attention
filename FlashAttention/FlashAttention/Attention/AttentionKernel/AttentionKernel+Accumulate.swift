//
//  AttentionKernel+Accumulate.swift
//  FlashAttention
//
//  Created by Philip Turner on 7/19/24.
//

// Operations where one operand is the attention matrix, the other operand is
// read from RAM.

// TODO: Refactor 'accumulate' to use the blocked algorithm.

struct AttentionAccumulateDescriptor {
  var index: String?
  var indexedBlockDimension: UInt16?
  var leadingBlockDimensionRHS: UInt16?
  var names: (accumulator: String, lhs: String, rhs: String)?
  var threadgroupAddress: String?
  var transposeStateRHS: Bool?
}

extension AttentionKernel {
  func accumulate(descriptor: AttentionAccumulateDescriptor) -> String {
    guard let index = descriptor.index,
          let indexedBlockDimension = descriptor.indexedBlockDimension,
          let leadingBlockDimensionRHS = descriptor.leadingBlockDimensionRHS,
          let names = descriptor.names,
          let threadgroupAddress = descriptor.threadgroupAddress,
          let transposeStateRHS = descriptor.transposeStateRHS else {
      fatalError("Descriptor was incomplete.")
    }
    
    return """
    
    // Where the did the async copy put the RHS?
    auto \(names.rhs)_block = (threadgroup float*)(\(threadgroupAddress));
    \(names.rhs)_block = simdgroup_matrix_storage<float>::apply_offset(
      \(names.rhs)_block,
      \(leadingBlockDimensionRHS),
      morton_offset,
      \(transposeStateRHS));
    
    // Iterate over the row/column dimension.
#pragma clang loop unroll(full)
    for (
      ushort \(index) = 0;
      \(index) < \(indexedBlockDimension);
      \(index) += 8
    ) {

      // Iterate over the head dimension.
#pragma clang loop unroll(full)
      for (ushort d = 0; d < \(paddedD); d += 8) {
        ushort2 origin(d, \(index));
        simdgroup_matrix_storage<float> \(names.rhs);
        
        // Load the RHS from threadgroup memory.
        \(names.rhs).load(
          \(names.rhs)_block,
          \(leadingBlockDimensionRHS),
          origin,
          \(transposeStateRHS));
        
        // Add the contributions from the c-th/r-th element of the attention
        // matrix row/column.
        \(names.accumulator)_sram[d / 8].multiply(
          \(names.lhs)_sram[\(index) / 8],
          \(names.rhs),
          /*accumulate=*/true);
      }
    }

"""
  }
}
