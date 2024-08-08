//
//  GEMMKernel+Multiply.swift
//  FlashAttention
//
//  Created by Philip Turner on 8/3/24.
//

extension GEMMKernel {
  struct MultiplyDescriptor {
    var addressSpace: String?
    var leadingDimensionA: String?
    var leadingDimensionB: String?
    var loadFunctionA: String?
    var loadFunctionB: String?
  }
  
  func createMultiply(descriptor: MultiplyDescriptor) -> String {
    guard let addressSpace = descriptor.addressSpace,
          let leadingDimensionA = descriptor.leadingDimensionA,
          let leadingDimensionB = descriptor.leadingDimensionB,
          let loadFunctionA = descriptor.loadFunctionA,
          let loadFunctionB = descriptor.loadFunctionB else {
      fatalError("Descriptor was incomplete.")
    }
    
    return """

// One multiply-accumulate loop iteration, or 8 dot products.
METAL_FUNC void multiply_accumulate(
const \(addressSpace) \(memoryName("A")) *A_src,
const \(addressSpace) \(memoryName("B")) *B_src,
thread simdgroup_matrix_storage<\(registerName("A"))> *A_sram,
thread simdgroup_matrix_storage<\(registerName("B"))> *B_sram,
thread simdgroup_matrix_storage<\(registerName("C"))> *C_sram,
ushort k
) {
#pragma clang loop unroll(full)
for (ushort m = 0; m < \(registerM); m += 8) {
  ushort2 origin(0, m);
  auto A = get_sram(A_sram, 8, origin);
  A->\(loadFunctionA)(A_src, \(leadingDimensionA), ushort2(k, m), A_trans);
}
#pragma clang loop unroll(full)
for (ushort n = 0; n < \(registerN); n += 8) {
  ushort2 origin(n, 0);
  auto B = get_sram(B_sram, \(registerN), origin);
  B->\(loadFunctionB)(B_src, \(leadingDimensionB), ushort2(n, k), B_trans);
}
#pragma clang loop unroll(full)
for (ushort m = 0; m < \(registerM); m += 8) {
#pragma clang loop unroll(full)
  for (ushort n = 0; n < \(registerN); n += 8) {
    auto A = get_sram(A_sram, 8, ushort2(0, m));
    auto B = get_sram(B_sram, \(registerN), ushort2(n, 0));
    auto C = get_sram(C_sram, \(registerN), ushort2(n, m));
    C->multiply(*A, *B);
  }
}
}

"""
  }
  
  func createUtilities() -> String {
    // Add the utility functions.
    var output = """

// Indexes into an array of registers.
//
// Calls to this function are expected to be evaluated at compile time. The
// array indices transform into register offsets, which are embedded into the
// assembly code.
template <typename T>
METAL_FUNC thread simdgroup_matrix_storage<T>* get_sram(
  thread simdgroup_matrix_storage<T> *sram,
  ushort sram_leading_dim,
  ushort2 matrix_origin
) {
  return sram + (matrix_origin.y / 8) * (sram_leading_dim / 8) + (matrix_origin.x / 8);
}
"""
    
    // Add the utility functions for the multiply-accumulate inner loop.
    do {
      var multiplyDesc = MultiplyDescriptor()
      if memoryPrecisions.A == .BF16, registerPrecisions.A == .FP32 {
        multiplyDesc.loadFunctionA = "load_bfloat"
      } else {
        multiplyDesc.loadFunctionA = "load"
      }
      if memoryPrecisions.B == .BF16, registerPrecisions.B == .FP32 {
        multiplyDesc.loadFunctionB = "load_bfloat"
      } else {
        multiplyDesc.loadFunctionB = "load"
      }
      
      multiplyDesc.addressSpace = "device"
      multiplyDesc.leadingDimensionA = leadingDimension("A")
      multiplyDesc.leadingDimensionB = leadingDimension("B")
      output += createMultiply(descriptor: multiplyDesc)
      
      multiplyDesc.addressSpace = "threadgroup"
      multiplyDesc.leadingDimensionA = "\(leadingBlockDimensions.A)"
      multiplyDesc.leadingDimensionB = "\(leadingBlockDimensions.B)"
      output += createMultiply(descriptor: multiplyDesc)
    }
    
    return output
  }
}

extension GEMMKernel {
  func createMultiplyIterations() -> String {
    var asyncIterationsStart: String
    if preferAsyncLoad {
      asyncIterationsStart = "0"
    } else {
      asyncIterationsStart = "(K - (K % K_group))"
    }
    let paddedCeilingK = "(K + K_remainder_padded - K_remainder)"
    
    return """

// Perform the iterations where async copy is avoided.
for (uint k = 0; k < \(asyncIterationsStart); k += 8) {
  uint2 A_offset(k, M_offset);
  uint2 B_offset(N_offset, k);
  A_offset += uint2(morton_offset.x, offset_in_group.y);
  B_offset += uint2(offset_in_group.x, morton_offset.y);
  
  auto A_src = simdgroup_matrix_storage<\(memoryName("A"))>::apply_offset(
    A, \(leadingDimension("A")), A_offset, A_trans);
  auto B_src = simdgroup_matrix_storage<\(memoryName("B"))>::apply_offset(
    B, \(leadingDimension("B")), B_offset, B_trans);

  simdgroup_matrix_storage<\(registerName("A"))> A_sram[
    \(registerM / 8) * (8 / 8)];
  simdgroup_matrix_storage<\(registerName("B"))> B_sram[
    (8 / 8) * \(registerN / 8)];
  multiply_accumulate(A_src, B_src,
                      A_sram, B_sram, C_sram, 0);
}

// Perform the iterations where async copy is used.
for (uint k = \(asyncIterationsStart); k < K; k += K_group) {
  auto A_block = (threadgroup \(memoryName("A"))*)(
    threadgroup_block);
  auto B_block = (threadgroup \(memoryName("B"))*)(
    threadgroup_block + \(blockBytes("A")));
  
  // Launch an async copy from device to threadgroup memory.
  if (sidx == 0) {
    uint2 A_offset(k, M_offset);
    uint2 B_offset(N_offset, k);
    auto A_src = simdgroup_matrix_storage<\(memoryName("A"))>::apply_offset(
      A, \(leadingDimension("A")), A_offset, A_trans);
    auto B_src = simdgroup_matrix_storage<\(memoryName("B"))>::apply_offset(
      B, \(leadingDimension("B")), B_offset, B_trans);

    ushort M_tile_dimension = min(uint(M_group), M - M_offset);
    ushort N_tile_dimension = min(uint(N_group), N - N_offset);
    ushort K_tile_dimension = min(uint(K_group), K - k);
    ushort K_tile_padded = min(uint(K_group), \(paddedCeilingK) - k);

    ushort2 A_tile_src(K_tile_dimension, M_tile_dimension);
    ushort2 B_tile_src(N_tile_dimension, K_tile_dimension);
    ushort2 A_tile_dst(K_tile_padded, M_tile_dimension);
    ushort2 B_tile_dst(N_tile_dimension, K_tile_padded);

    simdgroup_event events[2];
    events[0].async_copy(
      A_block, \(leadingBlockDimensions.A), A_tile_dst,
      A_src, \(leadingDimension("A")), A_tile_src, A_trans);
    events[1].async_copy(
      B_block, \(leadingBlockDimensions.B), B_tile_dst,
      B_src, \(leadingDimension("B")), B_tile_src, B_trans);
    simdgroup_event::wait(2, events);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  
  ushort2 A_block_offset(morton_offset.x, offset_in_group.y);
  ushort2 B_block_offset(offset_in_group.x, morton_offset.y);
  auto A_block_src = A_block;
  auto B_block_src = B_block;
  A_block_src = simdgroup_matrix_storage<\(memoryName("A"))>::apply_offset(
    A_block_src, \(leadingBlockDimensions.A), A_block_offset, A_trans);
  B_block_src = simdgroup_matrix_storage<\(memoryName("B"))>::apply_offset(
    B_block_src, \(leadingBlockDimensions.B), B_block_offset, B_trans);
  
  simdgroup_matrix_storage<\(registerName("A"))> A_sram[
    \(registerM / 8) * (K_group / 8)];
  simdgroup_matrix_storage<\(registerName("B"))> B_sram[
    (K_group / 8) * \(registerN / 8)];
#pragma clang loop unroll(full)
  for (ushort k = 0; k < K_remainder_padded; k += 8) {
    multiply_accumulate(A_block_src, B_block_src,
                        A_sram, B_sram, C_sram, k);
  }
  
  // Will there be any iterations after this one?
  if (k + K_group < K) {
    // If so, we haven't reached the edge of either input matrix yet.
#pragma clang loop unroll(full)
    for (ushort k = K_remainder_padded; k < K_group; k += 8) {
      multiply_accumulate(A_block_src, B_block_src,
                          A_sram, B_sram, C_sram, k);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

"""
  }
}
