//
//  AttentionKernel+Accumulate.swift
//  FlashAttention
//
//  Created by Philip Turner on 7/19/24.
//

// Operations where the LHS is the attention matrix.

// MARK: - Accumulate

struct AttentionAccumulateDescriptor {
  /// Name of left-hand side register allocation (32 x 32).
  var A: String?
  
  /// Name of right-hand side, source of a 32 x D block.
  var B: String?
  
  /// Name of product register allocation (32 x D).
  var C: String?
  var cacheC: Bool?
  
  var transposeState: (B: Bool, C: Bool)?
  var leadingDimensions: (B: String, C: String)?
  var matrixDimensions: (M: String, K: String)?
  var matrixOffset: (M: String, K: String)?
  
  /// Optional. Code to execute every time the accumulator is loaded.
  var everyIterationLoading: String?
  
  /// Optional. Code to only execute before storing the accumulator, on the
  /// last iteration of the K dimension.
  var lastIterationStoring: String?
}

extension AttentionKernel {
  func accumulate(descriptor: AttentionAccumulateDescriptor) -> String {
    guard let A = descriptor.A,
          let B = descriptor.B,
          let C = descriptor.C,
          let cacheC = descriptor.cacheC,
          let transposeState = descriptor.transposeState,
          let leadingDimensions = descriptor.leadingDimensions,
          let matrixDimensions = descriptor.matrixDimensions,
          let matrixOffset = descriptor.matrixOffset else {
      fatalError("Descriptor was incomplete.")
    }
    
    // Declare the block size along the D dimension.
    let blockDimensionD = 64
    
    var output: String = """

{
  const ushort D_block_dimension = \(blockDimensionD);
  
  // Declare the remainder of the row/column dimension.
  ushort K_remainder = (\(matrixDimensions.K) % 32 == 0)
    ? 32 : \(matrixDimensions.K) % 32;
  ushort K_remainder_padded = (K_remainder + 7) / 8 * 8;

"""
    
    if !cacheC {
      // 32 x 64 allocation in threadgroup memory
      // leading dimension = transposeC ? 32 : 64
      let leadingBlockDimensionC = transposeState.C 
      ? UInt16(32) : UInt16(blockDimensionD)
      
      output += """

const ushort \(C)_leading_block_dimension = \(leadingBlockDimensionC);

// Find where the \(C) data will be read from.
ushort2 \(C)_block_offset(morton_offset.x, morton_offset.y + sidx * 8);
auto \(C)_block = (threadgroup float*)(threadgroup_block);
\(C)_block = simdgroup_matrix_storage<float>::apply_offset(
  \(C)_block, \(C)_leading_block_dimension,
  \(C)_block_offset, \(transposeState.C));

"""
    }
    
    do {
      // 32 x 64 allocation in threadgroup memory
      // leading dimension = transposeB ? 32 : 64
      let leadingBlockDimensionB = transposeState.B 
      ? UInt16(32) : UInt16(blockDimensionD)
      
      output += """

const ushort \(B)_leading_block_dimension = \(leadingBlockDimensionB);

// Find where the \(B) data will be read from.
ushort2 \(B)_block_offset(morton_offset.x, morton_offset.y);
auto \(B)_block = (threadgroup float*)(threadgroup_block);
\(B)_block = simdgroup_matrix_storage<float>::apply_offset(
  \(B)_block, \(B)_leading_block_dimension,
  \(B)_block_offset, \(transposeState.B));

"""
    }
    
    output += """

// Outer loop over D.
#pragma clang loop unroll(\(cacheC ? "full" : "disable"))
for (ushort d_outer = 0; d_outer < D; d_outer += D_block_dimension) {
  ushort D_src_dimension = min(D_block_dimension, ushort(D - d_outer));
  ushort d_register_start = \(cacheC ? "d_outer" : "0");

"""

    if !cacheC {
      output += """

// Where the \(C) data will be written to.
simdgroup_matrix_storage<float> \(C)_sram[D_block_dimension / 8];

"""
    }
    
    output += """
    
if (\(matrixOffset.K) == 0) {
#pragma clang loop unroll(full)
  for (ushort d = 0; d < D_block_dimension; d += 8) {
    \(C)_sram[(d_register_start + d) / 8] =
    simdgroup_matrix_storage<float>(0);
  }
} else {

"""
    
    if !cacheC {
      let loopBody = """

ushort2 origin(d, 0);
\(C)_sram[(d_register_start + d) / 8].load(
  \(C)_block, \(C)_leading_block_dimension, origin, \(transposeState.C));

"""
      
      output += """

  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sidx == 0) {
    uint2 C_offset(d_outer, \(matrixOffset.M));
    auto src = simdgroup_matrix_storage<float>::apply_offset(
      \(C), \(leadingDimensions.C), C_offset, \(transposeState.C));
    auto dst = (threadgroup float*)(threadgroup_block);
    
    // It doesn't matter if the rows below the matrix edge are garbage.
    ushort M_src_dimension = min(
      uint(32), \(matrixDimensions.M) - \(matrixOffset.M));
    ushort2 tile_src(D_src_dimension, M_src_dimension);
    ushort2 tile_dst(D_src_dimension, M_src_dimension);
    
    simdgroup_event event;
    event.async_copy(
      dst, \(C)_leading_block_dimension, tile_dst,
      src, \(leadingDimensions.C), tile_src, \(transposeState.C));
    simdgroup_event::wait(1, &event);
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Iterate over the head dimension.
  if (D - d_outer >= D_block_dimension) {
  #pragma clang loop unroll(full)
    for (ushort d = 0; d < D_block_dimension; d += 8) {
      \(loopBody)
    }
  } else {
    #pragma clang loop unroll(full)
    for (ushort d = 0; d < D % D_block_dimension; d += 8) {
      \(loopBody)
    }
  }
  
"""
    }
    
    output += """
  
  \(descriptor.everyIterationLoading ?? "")
}

"""
    
    output += """

threadgroup_barrier(mem_flags::mem_threadgroup);
if (sidx == 0) {
  uint2 \(B)_offset(d_outer, \(matrixOffset.K));
  auto src = simdgroup_matrix_storage<float>::apply_offset(
    \(B), \(leadingDimensions.B), \(B)_offset, \(transposeState.B));
  auto dst = (threadgroup float*)(threadgroup_block);
  
  ushort K_src_dimension = min(
    uint(32), \(matrixDimensions.K) - \(matrixOffset.K));
  ushort K_dst_dimension = max(K_remainder_padded, K_src_dimension);
  ushort2 tile_src(D_src_dimension, K_src_dimension);
  ushort2 tile_dst(D_src_dimension, K_dst_dimension);
  
  simdgroup_event event;
  event.async_copy(
    dst, \(B)_leading_block_dimension, tile_dst,
    src, \(leadingDimensions.B), tile_src, \(transposeState.B));
  simdgroup_event::wait(1, &event);
}

"""
    
    let loopBodyAB = """

// Load the RHS from threadgroup memory.
ushort2 origin(d, k);
simdgroup_matrix_storage<float> \(B);
\(B).load(
  \(B)_block, \(B)_leading_block_dimension, origin, \(transposeState.B));

// Add the contributions from the c-th/r-th element of the
// attention matrix row/column.
\(C)_sram[(d_register_start + d) / 8].multiply(
  \(A)_sram[k / 8], \(B), /*accumulate=*/true);

"""
    
    let innerLoopAB = """

// Iterate over the head dimension.
if (D - d_outer >= D_block_dimension) {
#pragma clang loop unroll(full)
  for (ushort d = 0; d < D_block_dimension; d += 8) {
    \(loopBodyAB)
  }
} else {
#pragma clang loop unroll(full)
  for (ushort d = 0; d < D % D_block_dimension; d += 8) {
    \(loopBodyAB)
  }
}

"""
    
    output += """

threadgroup_barrier(mem_flags::mem_threadgroup);

// Iterate over the row/column dimension.
#pragma clang loop unroll(full)
for (ushort k = 0; k < K_remainder_padded; k += 8) {
  \(innerLoopAB)
}
if (\(matrixOffset.K) + 32 < \(matrixDimensions.K)) {
#pragma clang loop unroll(full)
  for (ushort k = K_remainder_padded; k < 32; k += 8) {
    \(innerLoopAB)
  }
} else {
  \(descriptor.lastIterationStoring ?? "")
}

"""
    if !cacheC {
      let loopBody = """

ushort2 origin(d, 0);
\(C)_sram[(d_register_start + d) / 8].store(
  \(C)_block, \(C)_leading_block_dimension, origin, \(transposeState.C));

"""
      
      output += """

threadgroup_barrier(mem_flags::mem_threadgroup);

// Iterate over the head dimension.
if (D - d_outer >= D_block_dimension) {
#pragma clang loop unroll(full)
  for (ushort d = 0; d < D_block_dimension; d += 8) {
    \(loopBody)
  }
} else {
#pragma clang loop unroll(full)
  for (ushort d = 0; d < D % D_block_dimension; d += 8) {
    \(loopBody)
  }
}

threadgroup_barrier(mem_flags::mem_threadgroup);
if (sidx == 0) {
  uint2 \(C)_offset(d_outer, \(matrixOffset.M));
  auto src = (threadgroup float*)(threadgroup_block);
  auto dst = simdgroup_matrix_storage<float>::apply_offset(
    \(C), \(leadingDimensions.C), \(C)_offset, \(transposeState.C));
  
  ushort D_dimension = min(ushort(D_block_dimension), ushort(D - d_outer));
  ushort M_src_dimension = min(
    uint(32), \(matrixDimensions.M) - \(matrixOffset.M));
  ushort2 tile_src(D_dimension, M_src_dimension);
  ushort2 tile_dst(D_dimension, M_src_dimension);
  
  simdgroup_event event;
  event.async_copy(
    dst, \(leadingDimensions.C), tile_dst,
    src, \(C)_leading_block_dimension, tile_src, \(transposeState.C));
  simdgroup_event::wait(1, &event);
}

"""
    }
    
    output += """

  }
}

"""
    
    return output
  }
}
