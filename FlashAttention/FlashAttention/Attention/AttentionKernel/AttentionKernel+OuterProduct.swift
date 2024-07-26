//
//  AttentionKernel+OuterProduct.swift
//  FlashAttention
//
//  Created by Philip Turner on 7/19/24.
//

// Operations where both arguments are read from RAM.

// MARK: - Two-Operand Access

struct AttentionTwoOperandAccessDescriptor {
  /// Name of left-hand side, source of a 32 x D block.
  var A: String?
  var cacheA: Bool?
  
  /// Name of right-hand side, source of a 32 x D block.
  var B: String?
  
  var transposeA: Bool?
  var transposeB: Bool?
  var leadingDimensionA: String?
  var leadingDimensionB: String?
  var matrixDimensions: (M: String, N: String)?
  var matrixOffset: (M: String, N: String)?
  
  /// Required. Code that sets the various pointers into threadgroup memory.
  var reservePointers: String?
  
  /// Optional. Loading code to only execute on the first D iteration.
  var firstIterationLoading: String?
  
  /// Required. Code for the inner loop, scoped over D.
  var innerLoop: String?
}

extension AttentionKernel {
  func twoOperandAccess(
    descriptor: AttentionTwoOperandAccessDescriptor
  ) -> String {
    guard let A = descriptor.A,
          let cacheA = descriptor.cacheA,
          let B = descriptor.B,
          let transposeA = descriptor.transposeA,
          let transposeB = descriptor.transposeB,
          let leadingDimensionA = descriptor.leadingDimensionA,
          let leadingDimensionB = descriptor.leadingDimensionB,
          let matrixDimensions = descriptor.matrixDimensions,
          let matrixOffset = descriptor.matrixOffset,
          let reservePointers = descriptor.reservePointers,
          let innerLoop = descriptor.innerLoop else {
      fatalError("Descriptor was incomplete.")
    }
    
    var output: String = """

{
  // Declare the remainder of the row/column dimension.
  ushort N_remainder = (\(matrixDimensions.N) % 32 == 0)
    ? 32 : \(matrixDimensions.N) % 32;
  ushort N_remainder_padded = (N_remainder + 7) / 8 * 8;

"""
    
    do {
      var leadingBlockDimensionB: UInt16
      var blockDimensionD: UInt16
      if cacheA {
        // 32 x 64 allocation in threadgroup memory
        // leading dimension = transposeB ? 32 : 64
        leadingBlockDimensionB = transposeB ? UInt16(32) : UInt16(64)
        blockDimensionD = 64
      } else {
        leadingBlockDimensionB = 32
        blockDimensionD = 32
      }
      
      output += """

const ushort \(B)_leading_block_dimension = \(leadingBlockDimensionB);
const ushort D_block_dimension = \(blockDimensionD);

"""
    }
    
    output += reservePointers
    
    // Outer loop over D.
    output += """
  
#pragma clang loop unroll(\(cacheA ? "full" : A == "V" ? "disable" : "disable"))
  for (ushort d_outer = 0; d_outer < D; d_outer += D_block_dimension) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    if (d_outer == 0) {
      \(descriptor.firstIterationLoading ?? "")
    }
    
    if (sidx == 0) {
      ushort D_src_dimension = min(D_block_dimension, ushort(D - d_outer));
      ushort D_dst_dimension = min(
        D_block_dimension, ushort(\(paddedD) - d_outer));
    
"""
    
    if cacheA {
      output += """
      
      uint2 \(B)_offset(d_outer, \(matrixOffset.N));
      auto src = simdgroup_matrix_storage<float>::apply_offset(
        \(B), \(leadingDimensionB), \(B)_offset, \(transposeB));
      auto dst = (threadgroup float*)(threadgroup_block);
      
      ushort N_src_dimension = min(
        uint(32), \(matrixDimensions.N) - \(matrixOffset.N));
      ushort N_dst_dimension = max(N_remainder_padded, N_src_dimension);
      ushort2 tile_src(D_src_dimension, N_src_dimension);
      ushort2 tile_dst(D_dst_dimension, N_dst_dimension);
      
      simdgroup_event event;
      event.async_copy(
        dst, \(B)_leading_block_dimension, tile_dst,
        src, \(leadingDimensionB), tile_src, \(transposeB));
      simdgroup_event::wait(1, &event);
      
"""
    } else {
      output += """
      
      // load \(A)[m]
      simdgroup_event events[2];
      {
        uint2 \(A)_offset(d_outer, \(matrixOffset.M));
        auto src = simdgroup_matrix_storage<float>::apply_offset(
          \(A), \(leadingDimensionA), \(A)_offset, \(transposeA));
        auto dst = (threadgroup float*)(threadgroup_block);
        
        ushort M_src_dimension = min(
          uint(32), \(matrixDimensions.M) - \(matrixOffset.M));
        ushort2 tile_src(D_src_dimension, M_src_dimension);
        ushort2 tile_dst(D_dst_dimension, M_src_dimension);
        events[0].async_copy(
          dst, 32, tile_dst,
          src, \(leadingDimensionA), tile_src, \(transposeA));
      }
      
      // load \(B)[n]
      {
        uint2 \(B)_offset(d_outer, \(matrixOffset.N));
        auto src = simdgroup_matrix_storage<float>::apply_offset(
          \(B), \(leadingDimensionB), \(B)_offset, \(transposeB));
        auto dst = (threadgroup float*)(threadgroup_block) + \(32 * 32);
        
        ushort N_src_dimension = min(
          uint(32), \(matrixDimensions.N) - \(matrixOffset.N));
        ushort N_dst_dimension = max(N_remainder_padded, N_src_dimension);
        ushort2 tile_src(D_src_dimension, N_src_dimension);
        ushort2 tile_dst(D_dst_dimension, N_dst_dimension);
        events[1].async_copy(
          dst, \(B)_leading_block_dimension, tile_dst,
          src, \(leadingDimensionB), tile_src, \(transposeB));
      }
      simdgroup_event::wait(2, events);
    
"""
    }
    
    output += """

    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    \(innerLoop)
  }
}

"""
    
    return output
  }
}

// MARK: - Outer Product

struct AttentionOuterProductDescriptor {
  /// Name of left-hand side, source of a 32 x D block.
  var A: String?
  var cacheA: Bool?
  
  /// Name of right-hand side, source of a 32 x D block.
  var B: String?
  
  /// Name of product register allocation (32 x 32).
  var C: String?
  
  var transposeA: Bool?
  var transposeB: Bool?
  var leadingDimensionA: String?
  var leadingDimensionB: String?
  var matrixDimensions: (M: String, N: String)?
  var matrixOffset: (M: String, N: String)?
}

extension AttentionKernel {
  // Accepts the operands A and B, then performs the multiplication A * B^T.
  //
  // A and C are divided along four SIMDs in the M dimension. Each SIMD carries
  // out an (8 x D) x (D x 32) matrix multiplication. The product has
  // dimensions 8 (M dimension) x 32 (N dimension). The caller specifies which
  // attention dimension (R, C) corresponds to N.
  //
  // Returns: Another descriptor, which you can intercept before materializing
  // the final source code.
  func outerProduct(
    descriptor: AttentionOuterProductDescriptor
  ) -> AttentionTwoOperandAccessDescriptor {
    guard let A = descriptor.A,
          let cacheA = descriptor.cacheA,
          let B = descriptor.B,
          let C = descriptor.C,
          let transposeA = descriptor.transposeA,
          let transposeB = descriptor.transposeB,
          let leadingDimensionA = descriptor.leadingDimensionA,
          let leadingDimensionB = descriptor.leadingDimensionB,
          let matrixDimensions = descriptor.matrixDimensions,
          let matrixOffset = descriptor.matrixOffset else {
      fatalError("Descriptor was incomplete.")
    }
    
    var accessDesc = AttentionTwoOperandAccessDescriptor()
    accessDesc.A = A
    accessDesc.cacheA = descriptor.cacheA
    accessDesc.B = B
    accessDesc.transposeA = transposeA
    accessDesc.transposeB = transposeB
    accessDesc.leadingDimensionA = leadingDimensionA
    accessDesc.leadingDimensionB = leadingDimensionB
    accessDesc.matrixDimensions = matrixDimensions
    accessDesc.matrixOffset = matrixOffset
    
    if cacheA {
      accessDesc.reservePointers = """
      
      // Find where the \(B) data will be read from.
      ushort2 \(B)_block_offset(morton_offset.x, morton_offset.y);
      auto \(B)T_block = (threadgroup float*)(threadgroup_block);
      \(B)T_block = simdgroup_matrix_storage<float>::apply_offset(
        \(B)T_block, \(B)_leading_block_dimension,
        \(B)_block_offset, \(!transposeB));

"""
    } else {
      accessDesc.reservePointers = """

      // Find where the \(A) data will be read from.
      ushort2 \(A)_block_offset(morton_offset.x, morton_offset.y + sidx * 8);
      auto \(A)_block = (threadgroup float*)(threadgroup_block);
      \(A)_block = simdgroup_matrix_storage<float>::apply_offset(
        \(A)_block, 32, \(A)_block_offset, \(transposeA));
      
      // Find where the \(B) data will be read from.
      ushort2 \(B)_block_offset(morton_offset.x, morton_offset.y);
      auto \(B)T_block = (threadgroup float*)(threadgroup_block) + \(32 * 32);
      \(B)T_block = simdgroup_matrix_storage<float>::apply_offset(
        \(B)T_block, 32, \(B)_block_offset, \(!transposeB));

"""
    }
    
    func innerLoopAB(startN: String, endN: String) -> String {
      var loopBody: String
      if cacheA {
        loopBody = """

// Inner loop over N.
#pragma clang loop unroll(full)
for (ushort n = \(startN); n < \(endN); n += 8) {
  // Load the RHS from threadgroup memory.
  ushort2 origin(n, d);
  simdgroup_matrix_storage<float> \(B)T;
  \(B)T.load(
    \(B)T_block, \(B)_leading_block_dimension, origin, \(!transposeB));
  
  // Mask out the first accumulate at compile-time.
  bool accumulate = (d_outer > 0) || (d > 0);
  \(C)_sram[n / 8].multiply(
    \(A)_sram[(d_outer + d) / 8], \(B)T, accumulate);
}

"""
      } else {
        loopBody = """

// Load the LHS from threadgroup memory.
ushort2 origin(d, 0);
simdgroup_matrix_storage<float> \(A);
\(A).load(\(A)_block, 32, origin, \(transposeA));

// Inner loop over N.
#pragma clang loop unroll(full)
for (ushort n = \(startN); n < \(endN); n += 8) {
  // Load the RHS from threadgroup memory.
  ushort2 origin(n, d);
  simdgroup_matrix_storage<float> \(B)T;
  \(B)T.load(\(B)T_block, 32, origin, \(!transposeB));
  
  // Mask out the first accumulate at compile-time.
  bool accumulate = (d_outer > 0) || (d > 0);
  \(C)_sram[n / 8].multiply(\(A), \(B)T, accumulate);
}

"""
      }
      
      return """

// Inner loop over D.
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
    
    accessDesc.innerLoop = """

// Iterate over the row/column dimension.
\(innerLoopAB(startN: "0", endN: "N_remainder_padded"))
if (\(matrixOffset.N) + 32 < \(matrixDimensions.N)) {
  \(innerLoopAB(startN: "N_remainder_padded", endN: "32"))
}

"""
    
    return accessDesc
  }
}
