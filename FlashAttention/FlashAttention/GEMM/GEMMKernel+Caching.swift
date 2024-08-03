//
//  GEMMKernel+Caching.swift
//  FlashAttention
//
//  Created by Philip Turner on 8/3/24.
//

extension GEMMKernel {
  func createLoadC() -> String {
    func loadAccumulator() -> String {
      return ""
    }
    
    return """
    
    simdgroup_matrix_storage<\(registerName("C"))> C_sram[
      \((registerM / 8) * (registerN / 8))];
    
    if (arguments.accumulateC) {
      \(loadAccumulator())
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
  
  func createStoreC() -> String {
    var storeFunctionC: String
    if memoryPrecisions.C == .BF16,
       registerPrecisions.C == .FP32 {
      storeFunctionC = "store_bfloat"
    } else {
      storeFunctionC = "store"
    }
    
    var condition: String
    if preferAsyncStore {
      condition = "false"
    } else {
      condition = "(M >= M_group) && (N >= N_group)"
    }
    
    return """

if (\(condition)) {
  // Fast path for matrices that qualify.
  uint2 C_offset(N_offset + offset_in_group.x,
                 M_offset + offset_in_group.y);
  auto C_dst = simdgroup_matrix_storage<\(memoryName("C"))>::apply_offset(
    C, N, C_offset);
  
  // Write the accumulator to device memory.
#pragma clang loop unroll(full)
  for (ushort m = 0; m < \(registerM); m += 8) {
#pragma clang loop unroll(full)
    for (ushort n = 0; n < \(registerN); n += 8) {
      ushort2 origin(n, m);
      auto C = get_sram(C_sram, \(registerN), origin);
      C->\(storeFunctionC)(C_dst, N, origin);
    }
  }
} else {
  // Slow path for when memory must be handled more carefully.
  auto C_block = (threadgroup \(memoryName("C"))*)(threadgroup_block);
  auto C_block_dst =
  simdgroup_matrix_storage<\(memoryName("C"))>::apply_offset(
    C_block, N_group, offset_in_group);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  
  // Write the accumulator to threadgroup memory.
#pragma clang loop unroll(full)
  for (ushort m = 0; m < \(registerM); m += 8) {
#pragma clang loop unroll(full)
    for (ushort n = 0; n < \(registerN); n += 8) {
      ushort2 origin(n, m);
      auto C = get_sram(C_sram, \(registerN), origin);
      C->\(storeFunctionC)(C_block_dst, N_group, origin);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  
  // Launch the async copy from threadgroup to device memory.
  if (sidx == 0) {
    uint2 C_offset(gid.x * N_group, gid.y * M_group);
    ushort2 C_tile(min(uint(N_group), N - C_offset.x),
                   min(uint(M_group), M - C_offset.y));
    auto C_dst = simdgroup_matrix_storage<\(memoryName("C"))>::apply_offset(
      C, N, C_offset);
    
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
        C_block, N_group, C_block_shift);
    }
    
    simdgroup_event event;
    event.async_copy(C_dst, N, C_tile, C_block, N_group, C_tile);
  }
}
"""
  }
}
