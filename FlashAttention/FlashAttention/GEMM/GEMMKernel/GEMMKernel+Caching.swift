//
//  GEMMKernel+Caching.swift
//  FlashAttention
//
//  Created by Philip Turner on 8/3/24.
//

extension GEMMKernel {
  // Whether the accumulator can be written directly to RAM.
  fileprivate var directAccessCondition: String {
    if preferAsyncStore {
      return "false"
    } else {
      return "(M >= M_group) && (N >= N_group)"
    }
  }
  
  func createInitializeC() -> String {
    """
    
    simdgroup_matrix_storage<\(registerName("C"))> C_sram[
      \((registerM / 8) * (registerN / 8))];
    
    if (arguments.accumulateC) {
      \(createLoadC())
    } else {
      #pragma clang loop unroll(full)
      for (ushort m = 0; m < \(registerM); m += 8) {
        #pragma clang loop unroll(full)
        for (ushort n = 0; n < \(registerN); n += 8) {
          ushort2 origin(n, m);
          auto C = get_sram(C_sram, \(registerN), origin);
          *C = simdgroup_matrix_storage<\(registerName("C"))>(0);
        }
      }
    }
    
    """
  }
  
  func createLoadC() -> String {
    var loadFunctionC: String
    if memoryPrecisions.C == .BF16,
       registerPrecisions.C == .FP32 {
      loadFunctionC = "load_bfloat"
    } else {
      loadFunctionC = "load"
    }
    
    return """

if (\(directAccessCondition) &&
    (gid.y * M_group < M_edge) &&
    (gid.x * N_group < N_edge)) {
  // Fast path for matrices that qualify.
  uint2 C_offset(N_offset + offset_in_group.x,
                 M_offset + offset_in_group.y);
  auto C_dst = simdgroup_matrix_storage<\(memoryName("C"))>::apply_offset(
    C, \(leadingDimension("C")), C_offset);
  
  // Write the accumulator to device memory.
#pragma clang loop unroll(full)
  for (ushort m = 0; m < \(registerM); m += 8) {
#pragma clang loop unroll(full)
    for (ushort n = 0; n < \(registerN); n += 8) {
      ushort2 origin(n, m);
      auto C = get_sram(C_sram, \(registerN), origin);
      C->\(loadFunctionC)(C_dst, \(leadingDimension("C")), origin);
    }
  }
} else {
  // Slow path for when memory must be handled more carefully.
  auto C_block = (threadgroup \(memoryName("C"))*)(threadgroup_block);
  auto C_block_dst =
  simdgroup_matrix_storage<\(memoryName("C"))>::apply_offset(
    C_block, \(leadingBlockDimensions.C), offset_in_group);
  
  // Launch the async copy from threadgroup to device memory.
  if (sidx == 0) {
    uint2 C_offset(N_offset, M_offset);
    ushort2 C_tile(min(uint(N_group), N - C_offset.x),
                   min(uint(M_group), M - C_offset.y));
    auto C_dst = simdgroup_matrix_storage<\(memoryName("C"))>::apply_offset(
      C, \(leadingDimension("C")), C_offset);
    
    simdgroup_event event;
    event.async_copy(
      C_block, \(leadingBlockDimensions.C), C_tile,
      C_dst, \(leadingDimension("C")), C_tile);
    simdgroup_event::wait(1, &event);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  
  // Read the accumulator from threadgroup memory.
#pragma clang loop unroll(full)
  for (ushort m = 0; m < \(registerM); m += 8) {
#pragma clang loop unroll(full)
    for (ushort n = 0; n < \(registerN); n += 8) {
      ushort2 origin(n, m);
      auto C = get_sram(C_sram, \(registerN), origin);
      C->\(loadFunctionC)(
        C_block_dst, \(leadingBlockDimensions.C), origin);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
}

"""
  }
  
  func createStoreC() -> String {
    var storeFunctionC: String
    if memoryPrecisions.C == .BF16,
       registerPrecisions.C == .FP32 {
      storeFunctionC = "store_bfloat"
    } else {
      storeFunctionC = "store"
    }
    
    return """

// Overlapping writes; this access must be sanitized through async copy.
//
// Either 'false' or
// 'directAccessCondition && (gid.y * M_group < M_edge) && (gid.x * N_group < N_edge)'.
if (\(directAccessCondition) &&
    (gid.y * M_group < M_edge) &&
    (gid.x * N_group < N_edge)) {
  // Fast path for matrices that qualify.
  uint2 C_offset(N_offset + offset_in_group.x,
                 M_offset + offset_in_group.y);
  auto C_dst = simdgroup_matrix_storage<\(memoryName("C"))>::apply_offset(
    C, \(leadingDimension("C")), C_offset);
  
  // Write the accumulator to device memory.
#pragma clang loop unroll(full)
  for (ushort m = 0; m < \(registerM); m += 8) {
#pragma clang loop unroll(full)
    for (ushort n = 0; n < \(registerN); n += 8) {
      ushort2 origin(n, m);
      auto C = get_sram(C_sram, \(registerN), origin);
      C->\(storeFunctionC)(C_dst, \(leadingDimension("C")), origin);
    }
  }
} else {
  // Slow path for when memory must be handled more carefully.
  auto C_block = (threadgroup \(memoryName("C"))*)(threadgroup_block);
  auto C_block_dst =
  simdgroup_matrix_storage<\(memoryName("C"))>::apply_offset(
    C_block, \(leadingBlockDimensions.C), offset_in_group);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  
  // Write the accumulator to threadgroup memory.
#pragma clang loop unroll(full)
  for (ushort m = 0; m < \(registerM); m += 8) {
#pragma clang loop unroll(full)
    for (ushort n = 0; n < \(registerN); n += 8) {
      ushort2 origin(n, m);
      auto C = get_sram(C_sram, \(registerN), origin);
      C->\(storeFunctionC)(
        C_block_dst, \(leadingBlockDimensions.C), origin);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  
  // Launch the async copy from threadgroup to device memory.
  if (sidx == 0) {
    uint2 C_offset(gid.x * N_group, gid.y * M_group);
    ushort2 C_tile(min(uint(N_group), N - C_offset.x),
                   min(uint(M_group), M - C_offset.y));
    auto C_dst = simdgroup_matrix_storage<\(memoryName("C"))>::apply_offset(
      C, \(leadingDimension("C")), C_offset);
    
    // If we shift successfully, the garbage zone moves from the bottom right
    // to the top left.
    if ((M_shift != 0) || (N_shift != 0)) {
      ushort2 C_block_shift(0, 0);
      if ((M_shift != 0) && (C_offset.y >= M_edge)) {
        C_block_shift.y = M_shift;
      }
      if ((N_shift != 0) && (C_offset.x >= N_edge)) {
        C_block_shift.x = N_shift;
      }
      C_block = simdgroup_matrix_storage<\(memoryName("C"))>::apply_offset(
        C_block, \(leadingBlockDimensions.C), C_block_shift);
    }
    
    simdgroup_event event;
    event.async_copy(
      C_dst, \(leadingDimension("C")), C_tile,
      C_block, \(leadingBlockDimensions.C), C_tile);
  }
}
"""
  }
}
