//
//  GEMM_ensemble.metal
//  MetalFlashAttention
//
//  Created by Philip Turner on 5/31/23.
//

#include "../Common/Common.hpp"
#include "../Common/SIMDFuturesPlaceholder.hpp"
#include "../Common/simdgroup_matrix_internals.hpp"
#include "Communication.hpp"
#include "GEMM.hpp"

#ifdef __METAL__

// Stream-K parallel decomposition (https://arxiv.org/abs/2301.03598).
#define STREAMK_PARALLEL_DECOMPOSITION 0

// Whether to enable fault counters. These prevent an infinite loop from
// occurring due to a semantic bug.
//
// TODO: Transform this into a function constant (part of the private API),
// strip the error code buffer binding.
#define STREAMK_FAULT_COUNTERS 1

#if STREAMK_FAULT_COUNTERS

#define FAULT_COUNTER_INIT(CODE) \
uint faultCounter##CODE = 0; \

#define FAULT_COUNTER_INCREMENT(CODE, TOLERANCE, ACTION) \
{ \
faultCounter##CODE += 1;\
\
if (faultCounter##CODE >= TOLERANCE) { \
errorCodes[gid] += faultCounter##CODE + CODE * 10 + 7; \
ACTION; \
} \
} \

#else

#define FAULT_COUNTER_INIT(CODE) \

#define FAULT_COUNTER_INCREMENT(CODE, TOLERANCE, ACTION) \

#endif

// This must be compiled with Xcode 14.2, as it uses SIMD futures on Apple 7
// devices. Other architectures can use different algorithms, selected through
// function constants.
//
// TODO: Support input values of varying types. The function's host name
// specifies the precision of the accumulator and result, but parameters are
// decided through function constants. This technique could also be applied to
// quantized GEMV, minimizing the number of shader permutations. When multiple
// input types are supported, you must allocate threadgroup memory in the
// encoder at runtime.
//
// TODO: Dynamically called Metal visible function that fuses the activation
// with the matrix multiplication. It should page the accumulator to threadgroup
// memory while calling into the function.
#if defined(__HAVE_SIMDGROUP_FUTURE__) || defined(SIMD_FUTURES_HIGHLIGHTING)
template <
typename real, // accumulator precision
typename atomic_real2_ptr,
ushort M_group, // threadgroup M
ushort N_group, // threadgroup N
ushort K_group, // threadgroup K
ushort M_simd, // simdgroup M
ushort N_simd, // simdgroup N
ushort K_simd // simdgroup K
>
void _gemm_streamk
 (
  device void* A [[buffer(0)]],
  device void* B [[buffer(1)]],
  device real* C [[buffer(2)]],
  threadgroup real* C_block [[threadgroup(0)]],
#if STREAMK_PARALLEL_DECOMPOSITION
  device atomic_uint* locks [[buffer(3)]],
  device void* partialSums [[buffer(4)]],
  device uint* errorCodes [[buffer(5)]],
  
  // Instantiate 'g' CTAs.
  uint grid_size [[threadgroups_per_grid]],
  uint gid [[threadgroup_position_in_grid]],
#else
  uint2 gid [[threadgroup_position_in_grid]],
#endif
  ushort sidx [[simdgroup_index_in_threadgroup]],
  ushort lane_id [[thread_index_in_simdgroup]])
{
  // Threadgroup MxNxK
  constexpr ulong2 A_tile(K_group, M_group);
  constexpr ulong2 B_tile(N_group, K_group);
  constexpr ushort A_block_size = M_group * K_group;
  threadgroup real* A_block = C_block + 0;
  threadgroup real* B_block = C_block + A_block_size;
  
  // TODO: Can we scope the tile allocations inside the MAC loop for Stream-K?
  //
  // Simdgroup MxNxK
  typedef vec<real, 2> real2;
  typedef simdgroup_matrix<real, K_simd, K_simd> simdgroup_real8x8;
  simdgroup_real8x8 A_value[M_simd / K_simd];
  simdgroup_real8x8 B_value[N_simd / K_simd];
  simdgroup_real8x8 C_value[M_simd / K_simd][N_simd / K_simd];
  
  // Arrange simds like:
  //
  //   N N N N     N N N N     N N N N
  // M 0 0 1 1   M 0 0 1 1   M 0 0 0 0
  // M 2 2 3 3   M 0 0 1 1   M 1 1 1 1
  // M 4 4 5 5   M 2 2 3 3   M 2 2 2 2
  // M 6 6 7 7   M 2 2 3 3   M 3 3 3 3
  constexpr ushort2 simds_per_group(N_group / N_simd, M_group / M_simd);
  ushort2 sid(sidx % simds_per_group.x, sidx / simds_per_group.x);
  
  ushort quad_id = lane_id / 4;
  ushort M_in_simd = get_simdgroup_matrix_m(lane_id, quad_id);
  ushort N_in_simd = get_simdgroup_matrix_n(lane_id, quad_id);
  
  // This thread is the only one that touches locks.
  ushort is_first_thread = (sidx == 0) && simd_is_first();
  
  // If unaligned, use a slower path when reading/writing.
  const bool aligned =
(  (M % M_group == 0) &&
  (N % N_group == 0) &&
 (K % K_group == 0) ) && false;
  
  // Currently, Stream-K only supports aligned matrices.
#if STREAMK_PARALLEL_DECOMPOSITION
  if (!aligned) {
    return;
  }
#endif
  
#if STREAMK_PARALLEL_DECOMPOSITION
  // Fail-safe in case you do something stupid while debugging.
  if (grid_size != num_threadgroups) {
    errorCodes[gid] = 42;
    return;
  }
  
  const uint M_tiles = (M + M_group - 1) / M_group;
  const uint N_tiles = (N + N_group - 1) / N_group;
  const uint K_tiles = (K + K_group - 1) / K_group;
  
  const uint iters_per_tile = K_tiles;
  uint total_iters = M_tiles * N_tiles * iters_per_tile;
  uint iters_per_cta = (total_iters + num_threadgroups - 1) / num_threadgroups;
  
  uint iter = gid * iters_per_cta;
  uint iter_end = min(iter + iters_per_cta, total_iters);
  if (iter >= total_iters) {
    return;
  }
  
  FAULT_COUNTER_INIT(1)
  while (iter < iter_end) {
    FAULT_COUNTER_INCREMENT(1, 10000, return);
    uint tile_id = iter / iters_per_tile;
    uint tile_iter = tile_id * iters_per_tile;
    uint tile_iter_end = tile_iter + iters_per_tile;
    
    // Perform the range of MAC iterations for this tile.
    uint local_iter = iter - tile_iter;
    uint local_iter_end = min(iter_end, tile_iter_end) - tile_iter;
    
    // Determine output tile coordinates.
    //
    // NOTE: The research paper's code can easily be misinterpreted.
    // Their indexing scheme was column-major; we use row-major.
    uint mm = M_group * (tile_id / N_tiles);
    uint nn = N_group * (tile_id % N_tiles);
    
    // Offsets of each matrix in memory.
    uint kk = local_iter * K_group;
    uint2 A_index(kk, mm);
    uint2 B_index(nn, kk);
#else
    uint2 A_index(0, gid.y * A_tile.y);
    uint2 B_index(gid.x * B_tile.x, 0);
#endif
    ushort2 C_offset(sid.x * N_simd + N_in_simd, sid.y * M_simd + M_in_simd);
    
    bool MN_store_full = aligned;
    ushort2 A_tile_variable;
    ushort2 B_tile_variable;
    
    // Short fast-path for outer simds.
    if (!aligned) {
      uint C_base_index_x = B_index.x + sid.x * N_simd;
      uint C_base_index_y = A_index.y + sid.y * M_simd;
      if (C_base_index_x >= N || C_base_index_y >= M) {
#if STREAMK_PARALLEL_DECOMPOSITION
        iter = tile_iter_end;
        continue;
#else
        return;
#endif
      }
      if (int(C_base_index_x) <= int(N - N_group) &&
          int(C_base_index_y) <= int(M - M_group)) {
        MN_store_full = true;
      }
    }
    
    const ushort k_edge = (K % K_group == 0) ? K_group : (K % K_group);
    
    // Rounded-up ceiling for the threadgroup block.
    const ushort k_elision_cutoff = (k_edge + K_simd - 1) / K_simd * K_simd;
    const uint k_elision_floor = K - k_edge;
    const uint k_elision_ceil = k_elision_floor + k_elision_cutoff;
    
    // Prefetch the first future.
    if (sidx == 0) {
      auto A_src = (device real*)A + A_index.y * K + A_index.x;
      auto B_src = (device real*)B + B_index.y * N + B_index.x;
      
      __metal_simdgroup_event_t events[2];
      if (!aligned) {
        A_tile_variable.y = min(uint(M_group), M - A_index.y);
        B_tile_variable.x = min(uint(N_group), N - B_index.x);
        
#if STREAMK_PARALLEL_DECOMPOSITION
        uint variable_k = min(uint(K_group), K - local_iter * K_group);
#else
        uint variable_k = min(uint(K_group), K - 0);
#endif
        
        // TODO: Use the element stride to emulate BF16. Read 2 bytes from
        // device memory, but add spacing when writing to threadgroup.
#define ASYNC_COPY_PART(K_FOR_PADDING) \
A_tile_variable.x = variable_k; \
B_tile_variable.y = variable_k; \
\
ushort padded_k; \
if (k_elision_cutoff == K_group) { \
padded_k = K_group; \
} else { \
padded_k = min(uint(K_group), k_elision_ceil - K_FOR_PADDING); \
} \
ushort2 A_tile_dst(padded_k, A_tile_variable.y); \
ushort2 B_tile_dst(B_tile_variable.x, padded_k); \
\
events[0] = __metal_simdgroup_async_copy_2d \
( \
sizeof(real), alignof(real), \
(threadgroup void *)(A_block), A_tile.x, 1, ulong2(A_tile_dst), \
(const device void *)(A_src), K, 1, ulong2(A_tile_variable), \
long2(0, 0), 0); \
events[1] = __metal_simdgroup_async_copy_2d \
( \
sizeof(real), alignof(real), \
(threadgroup void *)(B_block), B_tile.x, 1, ulong2(B_tile_dst), \
(const device void *)(B_src), N, 1, ulong2(B_tile_variable), \
long2(0, 0), 0); \
} else { \
events[0] = __metal_simdgroup_async_copy_2d \
( \
sizeof(real), alignof(real), \
(threadgroup void *)(A_block), A_tile.x, 1, A_tile, \
(const device void *)(A_src), K, 1, A_tile, \
long2(0, 0), 0); \
events[1] = __metal_simdgroup_async_copy_2d \
( \
sizeof(real), alignof(real), \
(threadgroup void *)(B_block), B_tile.x, 1, B_tile, \
(const device void *)(B_src), N, 1, B_tile, \
long2(0, 0), 0); \

          ASYNC_COPY_PART(0)
      }
      __metal_wait_simdgroup_events(2, events);
    }
    
    // Initialize the accumulator while waiting on the future's barrier.
    // TODO: Defer the preceding SIMD event wait in low-occupancy situations.
#pragma clang loop unroll(full)
    for (int m = 0; m < M_simd; m += K_simd) {
#pragma clang loop unroll(full)
      for (int n = 0; n < N_simd; n += K_simd) {
        C_value[m / K_simd][n / K_simd] = simdgroup_real8x8(0);
      }
    }
    
#if STREAMK_PARALLEL_DECOMPOSITION
    // TODO: Is the `min` on `k_end` really necessary?
    uint k_start = local_iter * K_group;
    uint k_end = min(K, local_iter_end * K_group);
#else
    uint k_start = 0;
    uint k_end = K;
#endif
    for (uint k_floor = k_start; k_floor < k_end; k_floor += K_group) {
      // The barrier needs to happen before computing the threadgroup pointer.
      threadgroup_barrier(mem_flags::mem_threadgroup);
      auto A_block_src = A_block + C_offset.y * K_group + N_in_simd;
      auto B_block_src = B_block + M_in_simd * N_group + C_offset.x;
      
      const ushort m_edge = (M % M_simd == 0) ? M_simd : (M % M_simd);
      const ushort n_edge = (N % N_simd == 0) ? N_simd : (N % N_simd);
      const ushort m_edge_roundup_8 = (m_edge + 7) / 8 * 8;
      const ushort n_edge_roundup_8 = (n_edge + 7) / 8 * 8;
      const ushort m_macc_end = M < M_simd ? m_edge_roundup_8 : M_simd;
      const ushort n_macc_end = N < N_simd ? n_edge_roundup_8 : N_simd;
      
#define A_LOAD_PART \
vec_load2< \
real, threadgroup real*, threadgroup real2*, ushort \
>(A_value[m / K_simd], A_block_src, A_tile.x, ushort2(0, m)); \

#define B_LOAD_PART \
vec_load2< \
real, threadgroup real*, threadgroup real2*, ushort \
>(B_value[n / K_simd], B_block_src, B_tile.x, ushort2(n, 0)); \

#define MACC_PART \
auto m_idx = m / K_simd; \
auto n_idx = n / K_simd; \
auto A = A_value[m_idx]; \
auto B = B_value[n_idx]; \
auto C = C_value[m_idx][n_idx]; \
simdgroup_multiply_accumulate(C_value[m_idx][n_idx], A, B, C); \

#define MACC_N_LOOP(START, END) \
for (ushort n = START; n < END; n += K_simd) { \
MACC_PART \
} \

#pragma clang loop unroll(full)
      for (ushort k = 0; k < k_elision_cutoff; k += K_simd) {
#pragma clang loop unroll(full)
        for (ushort m = 0; m < m_macc_end; m += K_simd) {
          A_LOAD_PART
        }
#pragma clang loop unroll(full)
        for (ushort n = 0; n < n_macc_end; n += K_simd) {
          B_LOAD_PART
        }
        A_block_src += K_simd;
        B_block_src += K_simd * N_group;
        
#pragma clang loop unroll(full)
        for (ushort m = 0; m < m_macc_end; m += K_simd) {
#pragma clang loop unroll(full)
          for (ushort n = 0; n < n_macc_end; n += K_simd) {
            MACC_PART
          }
        }
      }
      
      // Skip the threadgroup barrier on the last iteration.
      if (k_floor + K_group < k_end) {
#pragma clang loop unroll(full)
        for (ushort k = k_elision_cutoff; k < K_group; k += K_simd) {
#pragma clang loop unroll(full)
          for (ushort m = 0; m < m_macc_end; m += K_simd) {
            A_LOAD_PART
          }
#pragma clang loop unroll(full)
          for (ushort n = 0; n < n_macc_end; n += K_simd) {
            B_LOAD_PART
          }
          A_block_src += K_simd;
          B_block_src += K_simd * N_group;
          
#pragma clang loop unroll(full)
          for (ushort m = 0; m < m_macc_end; m += K_simd) {
#pragma clang loop unroll(full)
            for (ushort n = 0; n < n_macc_end; n += K_simd) {
              MACC_PART
            }
          }
        }
#undef A_LOAD_PART
#undef B_LOAD_PART
#undef MACC_PART
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        if (sidx == 0) {
          auto k_next = k_floor + K_group;
          auto A_src = (device real*)A + A_index.y * K + k_next;
          auto B_src = (device real*)B + k_next * N + B_index.x;
          
          __metal_simdgroup_event_t events[2];
          if (!aligned) {
            ushort variable_k = min(uint(K_group), k_end - k_next);
            ASYNC_COPY_PART(k_next)
#undef ASYNC_COPY_PART
          }
          __metal_wait_simdgroup_events(2, events);
        }
      }
    } // End MAC loop
    
#if STREAMK_PARALLEL_DECOMPOSITION
    // Consolidate partial-sums across CTAs.
    bool tile_started = (iter == tile_iter);
    bool tile_ended = (iter_end >= tile_iter_end);
    if (!tile_started) {
      // Store accum to temporary global storage.
      auto _partial = (device real*)partialSums;
      _partial += gid * (M_group * N_group);
      _partial += sidx * (M_simd * N_simd);
      _partial += M_in_simd * N_simd + N_in_simd;
      atomic_real2_ptr partial { _partial };
      
      FAULT_COUNTER_INIT(2)
      while (true) {
        FAULT_COUNTER_INCREMENT(2, 1000, return);
#pragma clang loop unroll(full)
        for (ushort m = 0; m < M_simd; m += K_simd) {
#pragma clang loop unroll(full)
          for (ushort n = 0; n < N_simd; n += K_simd) {
            partial.atomic_store
            (
             C_value[m / K_simd][n / K_simd], N_simd, ushort2(n, m));
          }
        }
        
#pragma clang loop unroll(full)
        for (ushort m = 0; m < M_simd; m += K_simd) {
          ushort successes[N_simd / K_simd];
#pragma clang loop unroll(full)
          for (ushort n = 0; n < N_simd; n += K_simd) {
            successes[n / K_simd] = partial.atomic_compare
            (
             C_value[m / K_simd][n / K_simd], N_simd, ushort2(n, m));
          }
          
          ushort succeeded = successes[0];
          for (short i = 0; i < N_simd / K_simd; ++i) {
            succeeded = succeeded && successes[i];
          }
          if (!succeeded) {
            continue;
          }
        }
        break;
      }
      threadgroup_barrier(mem_flags::mem_device);
      
      if (is_first_thread) {
        atomic_fetch_or_explicit(locks + gid, 1, RELAXED);
      }
    } else {
      // Store accumulators to output tile.
      if (!tile_ended) {
        // Accumulate partial sums from other CTA contributing to this tile.
        uint cta = gid + 1;
        uint cta_iter = iter_end;
        
        FAULT_COUNTER_INIT(3)
        while (cta_iter < tile_iter_end) {
          FAULT_COUNTER_INCREMENT(3, 10000, return)
          
          // TODO: Is this needed?
          threadgroup_barrier(mem_flags::mem_device);
          
          // Wait(flags[cta])
          auto exitEarly = (threadgroup bool*)C_block;
          if (is_first_thread) {
            auto lock = locks + cta;
            uint upperBits = atomic_load_explicit(lock, RELAXED);
            upperBits &= ~uint(1);
            uint desiredLock = upperBits + 2;
            
#if STREAMK_FAULT_COUNTERS
            *exitEarly = true;
#endif
            FAULT_COUNTER_INIT(4)
            while (true) {
              FAULT_COUNTER_INCREMENT(4, 1000, break)
              
              uint expectedLock = upperBits | 1;
              bool succeeded = atomic_compare_exchange_weak_explicit
              (
               lock, &expectedLock, upperBits | 1, RELAXED, RELAXED);
              if (succeeded) {
#if STREAMK_FAULT_COUNTERS
                *exitEarly = false;
#endif
                break;
              }
            }
            atomic_store_explicit(lock, desiredLock, RELAXED);
          }
#if STREAMK_FAULT_COUNTERS
          threadgroup_barrier(mem_flags::mem_threadgroup);
          if (*exitEarly) {
            return;
          }
#endif
          threadgroup_barrier(mem_flags::mem_device);
          
          // accum <- accum + LoadPartials(partials[cta])
          auto _partial = (device real*)partialSums;
          _partial += cta * (M_group * N_group);
          _partial += sidx * (M_simd * N_simd);
          _partial += M_in_simd * N_simd + N_in_simd;
          atomic_real2_ptr partial { _partial };
          
#pragma clang loop unroll(full)
          for (ushort m = 0; m < M_simd; m += K_simd) {
#pragma clang loop unroll(full)
            for (ushort n = 0; n < N_simd; n += K_simd) {
              partial.atomic_accumulate
              (
               C_value[m / K_simd][n / K_simd], N_simd, ushort2(n, m));
            }
          }
          
          // Increment counters for the next iteration.
          cta += 1;
          cta_iter += iters_per_cta;
        }
      } // if (!tile_ended)
#endif
      
      // Fast path that skips the async copy and writes directly to device
      // memory. This only provides a net performance gain when the matrix size
      // is divisible by 8.
      //
      // Small square matrices: GFLOPS (before -> after -> hybrid)
      // - Before: no direct store
      // - After: direct store everywhere (except when M % M_group == 0)
      // - Hybrid: direct store when (M % 8 == 0 and M % M_group != 0)
      //
      // M = 29 (15 -> 13 -> 15)
      // M = 30 (17 -> 15 -> 16)
      // M = 31 (19 -> 17 -> 19)
      // M = 32 (23 -> 24 -> 23)
      // M = 33 (18 -> 19 -> 18)
      // M = 34 (20 -> 22 -> 20)
      // M = 35 (23 -> 23 -> 23)
      // M = 36 (25 -> 26 -> 25)
      // M = 40 (35 -> 36 -> 36)
      // M = 48 (60 -> 62 -> 62)
      // M = 56 (83 -> 89 -> 89)
      // M = 61 (103 -> 93 -> 103)
      // M = 62 (109 -> 102 -> 109)
      // M = 63 (113 -> 103 -> 114)
      // M = 64 (135 -> 136 -> 136)
      // M = 65 (101 -> 101)
      // M = 66 (110 -> 109)
      const bool c_direct_store = (M % 8 == 0) && (N % 8 == 0);
      
      // StoreTile(C, tile_idx, accum)
      if (!aligned && c_direct_store) {
        auto C_index = uint2(B_index.x, A_index.y) + uint2(C_offset);
        auto C_dst = C + C_index.y * N + C_index.x;
        
#define STORE_PART \
vec_store2< \
real, device real*, device real2*, uint \
>(C_value[m / K_simd][n / K_simd], C_dst, N, ushort2(n, m)); \

#define STORE_PART_OVERSHOOT \
if (N_odd && N_in_simd == N_closest_even) { \
vec_store1< \
real, device real*, device real2*, uint \
>(C_value[m / K_simd][n / K_simd], C_dst, N, ushort2(n, m)); \
} else { \
STORE_PART \
} \
        
        // TODO: Only write SIMD matrices on the very, very edge to threadgroup
        // memory. That removes the need to subdivide the C cache among simds in
        // a threadgroup. Could the band be written with a single SIMD future
        // using the 'origin' parameter?
        const ushort m_edge = (M % M_simd == 0) ? M_simd : (M % M_simd);
        const ushort n_edge = (N % N_simd == 0) ? N_simd : (N % N_simd);
        const ushort m_edge_rounddown_8 = m_edge - m_edge % 8;
        const ushort n_edge_rounddown_8 = n_edge - n_edge % 8;
        const ushort m_overshoot = M % 8;
        const ushort n_overshoot = N % 8;
        
        const bool N_odd = (N % 2 != 0);
        const ushort N_closest_even = (N - N % 2) % 8;
        
        // Rounded-down ceiling for the accumulator.
        const uint m_elision_floor = M - m_edge;
        const uint n_elision_floor = N - n_edge;
        
        uint m_floor = C_index.y - C_index.y % 8;
        uint n_floor = C_index.x - C_index.x % 8;
        
        // Top left
#pragma clang loop unroll(full)
        for (ushort m = 0; m < m_edge_rounddown_8; m += K_simd) {
#pragma clang loop unroll(full)
          for (ushort n = 0; n < n_edge_rounddown_8; n += K_simd) {
            STORE_PART
          }
        }
        
        // Theory:
        //
        // The upper left is always filled. The outer edge (Y) is filled when
        // you haven't reached the last tile. The small strip (O) is filled when
        // you are at the last tile. The latter corresponds to the fraction of
        // the simd that can legally write to device memory.
        //
        // | <------------------- N_simd ---------------------> |
        // | <-- n_edge_rounddown_8 --> | n_overshoot |         |
        // | -------------------------- | ----------- | ------- |
        // |                            | OOOOOOOOOOO | YYYYYYY |
        // |                            | OOOOOOOOOOO | YYYYYYY |
        // |                            | OOOOOOOOOOO | YYYYYYY |
        // |                            | OOOOOOOOOOO | YYYYYYY |
        // |                            | OOOOOOOOOOO | YYYYYYY |
        // | -------------------------- | ----------- | ------- |
        // | OOOOOOOOOOOOOOOOOOOOOOOOOO | OOOOOOOOOOO | YYYYYYY |
        // | OOOOOOOOOOOOOOOOOOOOOOOOOO | OOOOOOOOOOO | YYYYYYY |
        // | OOOOOOOOOOOOOOOOOOOOOOOOOO | OOOOOOOOOOO | YYYYYYY |
        // | OOOOOOOOOOOOOOOOOOOOOOOOOO | OOOOOOOOOOO | YYYYYYY |
        // | OOOOOOOOOOOOOOOOOOOOOOOOOO | OOOOOOOOOOO | YYYYYYY |
        // | OOOOOOOOOOOOOOOOOOOOOOOOOO | OOOOOOOOOOO | YYYYYYY |
        // | OOOOOOOOOOOOOOOOOOOOOOOOOO | OOOOOOOOOOO | YYYYYYY |
        // | -------------------------- | ----------- | ------- |
        // | YYYYYYYYYYYYYYYYYYYYYYYYYY | YYYYYYYYYYY | YYYYYYY |
        // | -------------------------- | ----------- | ------- |
        //
        // There's a problem. What happens when you're at (Y) in one dimension,
        // but (O) in another? We must partition the output matrix into 9 zones,
        // which are filled based on a set of conditions.
        // - Y blocks may only be touched by writes before the last matrix tile
        // - O blocks may be touched by anything up to/including the last tile
        //
        // | ---------- | ------ | ------ |
        // | Upper Left | Zone B | Zone C |
        // | ---------- | ------ | ------ |
        // | Zone D     | Zone E | Zone F |
        // | ---------- | ------ | ------ |
        // | Zone G     | Zone H | Zone I |
        // | ---------- | ------ | ------ |
        
        const bool m_aligned = (M % 8 == 0);
        const bool n_aligned = (N % 8 == 0);
        bool is_zone_b = (N_in_simd < n_overshoot);
        bool is_zone_c = (n_floor < n_elision_floor);
        bool is_zone_d = (M_in_simd < m_overshoot);
        bool is_zone_g = (m_floor < m_elision_floor);
        
        if (m_edge != M_simd) {
          if (is_zone_g) {
#pragma clang loop unroll(full)
            for (ushort m = m_edge_rounddown_8; m < M_simd; m += K_simd) {
#pragma clang loop unroll(full)
              for (ushort n = 0; n < n_edge_rounddown_8; n += K_simd) {
                STORE_PART
              }
            }
          } else if (!m_aligned) {
            ushort m = m_edge_rounddown_8;
            if (is_zone_d) {
#pragma clang loop unroll(full)
              for (ushort n = 0; n < n_edge_rounddown_8; n += K_simd) {
                STORE_PART
              }
              if (is_zone_c) {
#pragma clang loop unroll(full)
                for (ushort n = n_edge_rounddown_8; n < N_simd; n += K_simd) {
                  STORE_PART
                }
              }
            }
          }
        }
        
        if (n_edge != N_simd) {
          if (is_zone_c) {
#pragma clang loop unroll(full)
            for (ushort m = 0; m < m_edge_rounddown_8; m += K_simd) {
#pragma clang loop unroll(full)
              for (ushort n = n_edge_rounddown_8; n < N_simd; n += K_simd) {
                STORE_PART
              }
            }
          } else if (!n_aligned) {
            ushort n = n_edge_rounddown_8;
            if (is_zone_b) {
#pragma clang loop unroll(full)
              for (ushort m = 0; m < m_edge_rounddown_8; m += K_simd) {
                STORE_PART_OVERSHOOT
              }
              if (is_zone_g) {
#pragma clang loop unroll(full)
                for (ushort m = m_edge_rounddown_8; m < M_simd; m += K_simd) {
                  STORE_PART_OVERSHOOT
                }
              }
            }
          }
        }

        if ((m_edge != M_simd) && (n_edge != N_simd)) {
          if (is_zone_c && is_zone_g) {
#pragma clang loop unroll(full)
            for (ushort m = m_edge_rounddown_8; m < M_simd; m += K_simd) {
#pragma clang loop unroll(full)
              for (ushort n = n_edge_rounddown_8; n < N_simd; n += K_simd) {
                STORE_PART
              }
            }
          } else if (!m_aligned && !n_aligned) {
            if (is_zone_b && is_zone_d) {
              ushort m = m_edge_rounddown_8;
              ushort n = n_edge_rounddown_8;
              STORE_PART_OVERSHOOT
            }
          }
        }
#undef STORE_PART
#undef STORE_PART_OVERSHOOT
      } else if (aligned) {
        auto C_index = uint2(B_index.x, A_index.y) + uint2(C_offset);
        auto C_dst = C + C_index.y * N + C_index.x;
        
#pragma clang loop unroll(full)
        for (ushort m = 0; m < M_simd; m += K_simd) {
#pragma clang loop unroll(full)
          for (ushort n = 0; n < N_simd; n += K_simd) {
            vec_store2<
            real, device real*, device real2*, uint
            >(C_value[m / K_simd][n / K_simd], C_dst, N, ushort2(n, m));
          }
        }
      } else {
        // - Algorithm 1: Each simd takes turns writing to memory
        // - Algorithm 2: All simds write small chunks at the same time
#define SIMD_ASYNC_STORE_ALGORITHM 2
        constexpr ushort B_block_size = K_group * N_group;
        constexpr ushort C_block_size = A_block_size + B_block_size;
        constexpr ushort simds_total = (M_group / M_simd) * (N_group / N_simd);
        
        // Requires that blocks are large enough to fit at least one simd.
#if SIMD_ASYNC_STORE_ALGORITHM == 1
        constexpr ushort C_size_simd = M_simd * N_simd;
        constexpr ushort simds_per_iter = C_block_size / C_size_simd;
        
        // Ensure the row size isn't zero (the compiler will throw an error from
        // allocating a zero-length array).
        float _assert_rows_per_simd_nonzero[simds_per_iter];
        
        ushort active_sidx = 0;
        for (; active_sidx < simds_total; active_sidx += simds_per_iter) {
          threadgroup_barrier(mem_flags::mem_threadgroup);
          if (active_sidx <= sidx && sidx < active_sidx + simds_per_iter) {
            auto C_src = C_block + (sidx - active_sidx) * C_size_simd;
            C_src += M_in_simd * N_simd + N_in_simd;
            uint2 C_tile(N_simd, M_simd);
            
#pragma clang loop unroll(full)
            for (ushort m = 0; m < M_simd; m += K_simd) {
              ushort m_in_block = m;
#if 0
              // These three closing braces are necessary to stop Xcode's
              // auto-indentation from messing up the code.
            }
          }
        }
#endif
#endif // SIMD_ASYNC_STORE_ALGORITHM == 1
        
        // Requires that blocks have at least `8 * simds_total` rows. There are
        // also some edge cases (not worth investigating) that cause incorrect
        // behavior. These can be fixed by avoiding that specific block size.
#if SIMD_ASYNC_STORE_ALGORITHM == 2
        constexpr ushort C_row_size = K_simd * N_simd;
        constexpr ushort _r = K_simd * C_block_size / C_row_size / simds_total;
        constexpr ushort _rows = (_r > M_simd) ? M_simd : _r;
        constexpr ushort rows_per_simd = _rows / K_simd;
        constexpr ushort loop_row_jump = K_simd * rows_per_simd;
        
        // Ensure the row size isn't zero (the compiler will throw an error from
        // allocating a zero-length array).
        float _assert_rows_per_simd_nonzero[rows_per_simd];
        
        auto C_src = C_block + (sidx * rows_per_simd * C_row_size);
        C_src += M_in_simd * N_simd + N_in_simd;
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        {
#pragma clang loop unroll(full)
          for (ushort m_floor = 0; m_floor < M_simd; m_floor += loop_row_jump) {
            ushort2 C_tile(N_simd, loop_row_jump);
            if (M_simd % rows_per_simd != 0) {
              C_tile.y = min(C_tile.y, ushort(M_simd - m_floor));
            }
#pragma clang loop unroll(full)
            for (ushort _m = 0; _m < C_tile.y; _m += K_simd) {
              ushort m_in_block = _m;
              ushort m = m_floor + _m;
#endif // SIMD_ASYNC_STORE_ALGORITHM == 2
              
#pragma clang loop unroll(full)
              for (ushort n = 0; n < N_simd; n += K_simd) {
                ushort2 origin(n, m_in_block);
                vec_store2<
                real, threadgroup real*, threadgroup real2*, ushort
                >(C_value[m / K_simd][n / K_simd], C_src, N_simd, origin);
              }
            }
            
            auto C_index = uint2(B_index.x, A_index.y) + uint2(C_offset);
            auto C_dst = C + C_index.y * N + C_index.x;
#if SIMD_ASYNC_STORE_ALGORITHM == 2
            C_offset.y += loop_row_jump;
            if (loop_row_jump < M_simd) {
              simdgroup_barrier(mem_flags::mem_threadgroup);
            }
#endif
            
            bool MN_store_full = false;
            if (!aligned) {
              if (int(C_index.x) <= int(N - N_group) &&
                  int(C_index.y) <= int(M - M_group)) {
                MN_store_full = true;
              }
              MN_store_full = bool(simd_broadcast_first(int(MN_store_full)));
            }
            
            // Only checks pointer values for the first lane; most threads have
            // garbage values and should theoretically cause undefined behavior.
            __metal_simdgroup_event_t event;
            if (!aligned && !MN_store_full) {
              uint2 C_tile_variable;
              C_tile_variable.x = min(uint(C_tile.x), N - C_index.x);
              C_tile_variable.y = min(uint(C_tile.y), M - C_index.y);
              
              event = __metal_simdgroup_async_copy_2d
              (
               sizeof(real), alignof(real),
               (device void *)(C_dst), N, 1, ulong2(C_tile_variable),
               (const threadgroup void *)(C_src), N_simd, 1, ulong2(C_tile),
               long2(0, 0), 0);
            } else {
              event = __metal_simdgroup_async_copy_2d
              (
               sizeof(real), alignof(real),
               (device void *)(C_dst), N, 1, ulong2(C_tile),
               (const threadgroup void *)(C_src), N_simd, 1, ulong2(C_tile),
               long2(0, 0), 0);
            }
            __metal_wait_simdgroup_events(1, &event);
          }
        }
      } // if (aligned)
#if STREAMK_PARALLEL_DECOMPOSITION
    } // if (!tile_started)
    iter = tile_iter_end;
  }
#endif
}

#if STREAMK_PARALLEL_DECOMPOSITION
    
#define MAKE_GEMM_STREAMK(a, b) \
if (simds_per_threadgroup.y == a && \
simds_per_threadgroup.x == b) { \
constexpr ushort M_splits = a; \
constexpr ushort N_splits = b; \
\
_gemm_streamk< \
real, \
atomic_real2_ptr, \
M_group, \
N_group, \
K_group, \
M_group / M_splits, \
N_group / N_splits, \
K_simd \
>( \
A, B, C, C_block, locks, partialSums, errorCodes, grid_size, gid, sidx, \
lane_id);\
} \
    
#else

#define MAKE_GEMM_STREAMK(a, b) \
if (simds_per_threadgroup.y == a && \
simds_per_threadgroup.x == b) { \
constexpr ushort M_splits = a; \
constexpr ushort N_splits = b; \
\
_gemm_streamk< \
real, \
atomic_real2_ptr, \
M_group, \
N_group, \
K_group, \
M_group / M_splits, \
N_group / N_splits, \
K_simd \
>(A, B, C, C_block, gid, sidx, lane_id); \
} \

#endif

// TODO: Fix the function signatures to the pointer appears before the letter,
// like `device void *A`.

kernel void hgemm
 (
  device void* A [[buffer(0)]],
  device void* B [[buffer(1)]],
  device half* C [[buffer(2)]],
#if STREAMK_PARALLEL_DECOMPOSITION
  device atomic_uint* locks [[buffer(3)]],
  device void* partialSums [[buffer(4)]],
  device uint* errorCodes [[buffer(5)]],
  
  // Instantiate 'g' CTAs.
  uint grid_size [[threadgroups_per_grid]],
  uint gid [[threadgroup_position_in_grid]],
#else
  uint2 gid [[threadgroup_position_in_grid]],
#endif
  ushort sidx [[simdgroup_index_in_threadgroup]],
  ushort lane_id [[thread_index_in_simdgroup]])
{
  typedef half real;
  typedef atomic_half2_ptr atomic_real2_ptr;
  
  constexpr ushort M_group = 32; // threadgroup M
  constexpr ushort N_group = 32; // threadgroup N
  constexpr ushort K_group = 32; // threadgroup K
  
  constexpr ushort K_simd = 8; // simdgroup K
  constexpr ushort A_block_size = M_group * K_group;
  constexpr ushort B_block_size = K_group * N_group;
  
  threadgroup real C_block[A_block_size + B_block_size];
  
  _gemm_streamk<
  real,
  atomic_real2_ptr,
  M_group,
  N_group,
  K_group,
  M_group / 2,
  N_group / 2,
  K_simd
  >(A, B, C, C_block, gid, sidx, lane_id);
}

kernel void hgemm_16x64
 (
  device void* A [[buffer(0)]],
  device void* B [[buffer(1)]],
  device half* C [[buffer(2)]],
#if STREAMK_PARALLEL_DECOMPOSITION
  device atomic_uint* locks [[buffer(3)]],
  device void* partialSums [[buffer(4)]],
  device uint* errorCodes [[buffer(5)]],
  
  // Instantiate 'g' CTAs.
  uint grid_size [[threadgroups_per_grid]],
  uint gid [[threadgroup_position_in_grid]],
#else
  uint2 gid [[threadgroup_position_in_grid]],
#endif
  ushort sidx [[simdgroup_index_in_threadgroup]],
  ushort lane_id [[thread_index_in_simdgroup]])
{
  typedef half real;
  typedef atomic_half2_ptr atomic_real2_ptr;
  
  constexpr ushort M_group = 16; // threadgroup M
  constexpr ushort N_group = 16; // threadgroup N
  constexpr ushort K_group = 64; // threadgroup K
  
  constexpr ushort K_simd = 8; // simdgroup K
  constexpr ushort A_block_size = M_group * K_group;
  constexpr ushort B_block_size = K_group * N_group;
  
  threadgroup real C_block[A_block_size + B_block_size];
  
  _gemm_streamk<
  real,
  atomic_real2_ptr,
  M_group,
  N_group,
  K_group,
  M_group / 2,
  N_group / 2,
  K_simd
  >(A, B, C, C_block, gid, sidx, lane_id);
}

kernel void hgemm_16x64_batched
 (
  device void* A [[buffer(0)]],
  device void* B [[buffer(1)]],
  device half* C [[buffer(2)]],
  constant ulong4 *ABC_offsets [[buffer(3)]],
#if STREAMK_PARALLEL_DECOMPOSITION
  device atomic_uint* locks [[buffer(3)]],
  device void* partialSums [[buffer(4)]],
  device uint* errorCodes [[buffer(5)]],
  
  // Instantiate 'g' CTAs.
  uint grid_size [[threadgroups_per_grid]],
  uint gid [[threadgroup_position_in_grid]],
#else
  uint3 gid [[threadgroup_position_in_grid]],
#endif
  ushort sidx [[simdgroup_index_in_threadgroup]],
  ushort lane_id [[thread_index_in_simdgroup]])
{
  typedef half real;
  typedef atomic_half2_ptr atomic_real2_ptr;
  
  constexpr ushort M_group = 16; // threadgroup M
  constexpr ushort N_group = 16; // threadgroup N
  constexpr ushort K_group = 64; // threadgroup K
  
  constexpr ushort K_simd = 8; // simdgroup K
  constexpr ushort A_block_size = M_group * K_group;
  constexpr ushort B_block_size = K_group * N_group;
  
  threadgroup real C_block[A_block_size + B_block_size];
  
  ulong3 offsets = ABC_offsets[gid.z].xyz;
  auto _A = (device half*)A + offsets[0];
  auto _B = (device half*)B + offsets[1];
  auto _C = (device half*)C + offsets[2];
  
  _gemm_streamk<
  real,
  atomic_real2_ptr,
  M_group,
  N_group,
  K_group,
  M_group / 2,
  N_group / 2,
  K_simd
  >(_A, _B, _C, C_block, gid.xy, sidx, lane_id);
}

kernel void hgemm_32x32
 (
  device void* A [[buffer(0)]],
  device void* B [[buffer(1)]],
  device half* C [[buffer(2)]],
#if STREAMK_PARALLEL_DECOMPOSITION
  device atomic_uint* locks [[buffer(3)]],
  device void* partialSums [[buffer(4)]],
  device uint* errorCodes [[buffer(5)]],
  
  // Instantiate 'g' CTAs.
  uint grid_size [[threadgroups_per_grid]],
  uint gid [[threadgroup_position_in_grid]],
#else
  uint2 gid [[threadgroup_position_in_grid]],
#endif
  ushort sidx [[simdgroup_index_in_threadgroup]],
  ushort lane_id [[thread_index_in_simdgroup]])
{
  typedef half real;
  typedef atomic_half2_ptr atomic_real2_ptr;
  
  constexpr ushort M_group = 32; // threadgroup M
  constexpr ushort N_group = 32; // threadgroup N
  constexpr ushort K_group = 32; // threadgroup K
  
  constexpr ushort K_simd = 8; // simdgroup K
  constexpr ushort A_block_size = M_group * K_group;
  constexpr ushort B_block_size = K_group * N_group;
  
  threadgroup real C_block[A_block_size + B_block_size];
  
  _gemm_streamk<
  real,
  atomic_real2_ptr,
  M_group,
  N_group,
  K_group,
  M_group / 2,
  N_group / 2,
  K_simd
  >(A, B, C, C_block, gid, sidx, lane_id);
}

kernel void hgemm_32x32_batched
 (
  device void* A [[buffer(0)]],
  device void* B [[buffer(1)]],
  device half* C [[buffer(2)]],
  constant ulong4 *ABC_offsets [[buffer(3)]],
#if STREAMK_PARALLEL_DECOMPOSITION
  device atomic_uint* locks [[buffer(3)]],
  device void* partialSums [[buffer(4)]],
  device uint* errorCodes [[buffer(5)]],
  
  // Instantiate 'g' CTAs.
  uint grid_size [[threadgroups_per_grid]],
  uint gid [[threadgroup_position_in_grid]],
#else
  uint3 gid [[threadgroup_position_in_grid]],
#endif
  ushort sidx [[simdgroup_index_in_threadgroup]],
  ushort lane_id [[thread_index_in_simdgroup]])
{
  typedef half real;
  typedef atomic_half2_ptr atomic_real2_ptr;
  
  constexpr ushort M_group = 32; // threadgroup M
  constexpr ushort N_group = 32; // threadgroup N
  constexpr ushort K_group = 32; // threadgroup K
  
  constexpr ushort K_simd = 8; // simdgroup K
  constexpr ushort A_block_size = M_group * K_group;
  constexpr ushort B_block_size = K_group * N_group;
  
  threadgroup real C_block[A_block_size + B_block_size];
  
  ulong3 offsets = ABC_offsets[gid.z].xyz;
  auto _A = (device half*)A + offsets[0];
  auto _B = (device half*)B + offsets[1];
  auto _C = (device half*)C + offsets[2];
  
  _gemm_streamk<
  real,
  atomic_real2_ptr,
  M_group,
  N_group,
  K_group,
  M_group / 2,
  N_group / 2,
  K_simd
  >(_A, _B, _C, C_block, gid.xy, sidx, lane_id);
}

kernel void hgemm_48x32
 (
  device void* A [[buffer(0)]],
  device void* B [[buffer(1)]],
  device half* C [[buffer(2)]],
#if STREAMK_PARALLEL_DECOMPOSITION
  device atomic_uint* locks [[buffer(3)]],
  device void* partialSums [[buffer(4)]],
  device uint* errorCodes [[buffer(5)]],
  
  // Instantiate 'g' CTAs.
  uint grid_size [[threadgroups_per_grid]],
  uint gid [[threadgroup_position_in_grid]],
#else
  uint2 gid [[threadgroup_position_in_grid]],
#endif
  ushort sidx [[simdgroup_index_in_threadgroup]],
  ushort lane_id [[thread_index_in_simdgroup]])
{
  typedef half real;
  typedef atomic_half2_ptr atomic_real2_ptr;
  
  constexpr ushort M_group = 48; // threadgroup M
  constexpr ushort N_group = 48; // threadgroup N
  constexpr ushort K_group = 32; // threadgroup K
  
  constexpr ushort K_simd = 8; // simdgroup K
  constexpr ushort A_block_size = M_group * K_group;
  constexpr ushort B_block_size = K_group * N_group;
  
  threadgroup real C_block[A_block_size + B_block_size];
  
  _gemm_streamk<
  real,
  atomic_real2_ptr,
  M_group,
  N_group,
  K_group,
  M_group / 2,
  N_group / 2,
  K_simd
  >(A, B, C, C_block, gid, sidx, lane_id);
}

kernel void hgemm_48x32_batched
 (
  device void* A [[buffer(0)]],
  device void* B [[buffer(1)]],
  device half* C [[buffer(2)]],
  constant ulong4 *ABC_offsets [[buffer(3)]],
#if STREAMK_PARALLEL_DECOMPOSITION
  device atomic_uint* locks [[buffer(3)]],
  device void* partialSums [[buffer(4)]],
  device uint* errorCodes [[buffer(5)]],
  
  // Instantiate 'g' CTAs.
  uint grid_size [[threadgroups_per_grid]],
  uint gid [[threadgroup_position_in_grid]],
#else
  uint3 gid [[threadgroup_position_in_grid]],
#endif
  ushort sidx [[simdgroup_index_in_threadgroup]],
  ushort lane_id [[thread_index_in_simdgroup]])
{
  typedef half real;
  typedef atomic_half2_ptr atomic_real2_ptr;
  
  constexpr ushort M_group = 48; // threadgroup M
  constexpr ushort N_group = 48; // threadgroup N
  constexpr ushort K_group = 32; // threadgroup K
  
  constexpr ushort K_simd = 8; // simdgroup K
  constexpr ushort A_block_size = M_group * K_group;
  constexpr ushort B_block_size = K_group * N_group;
  
  threadgroup real C_block[A_block_size + B_block_size];
  
  ulong3 offsets = ABC_offsets[gid.z].xyz;
  auto _A = (device half*)A + offsets[0];
  auto _B = (device half*)B + offsets[1];
  auto _C = (device half*)C + offsets[2];
  
  _gemm_streamk<
  real,
  atomic_real2_ptr,
  M_group,
  N_group,
  K_group,
  M_group / 2,
  N_group / 2,
  K_simd
  >(_A, _B, _C, C_block, gid.xy, sidx, lane_id);
}

// hgemm_batched

#ifdef MFA_MONOLITHIC_KERNELS

// Part of the public API, but doesn't require block size to be set through
// function constants.
kernel void hgemm_monolithic
 (
  device void* A [[buffer(0)]],
  device void* B [[buffer(1)]],
  device half* C [[buffer(2)]],
  uint2 gid [[threadgroup_position_in_grid]],
  ushort sidx [[simdgroup_index_in_threadgroup]],
  ushort lane_id [[thread_index_in_simdgroup]])
{
  typedef half real;
  typedef atomic_half2_ptr atomic_real2_ptr;
  
  constexpr ushort M_group = 32; // threadgroup M
  constexpr ushort N_group = 32; // threadgroup N
  constexpr ushort K_group = 32; // threadgroup K
  
  constexpr ushort K_simd = 8; // simdgroup K
  constexpr ushort A_block_size = M_group * K_group;
  constexpr ushort B_block_size = K_group * N_group;
  
  threadgroup real C_block[A_block_size + B_block_size];
  
  _gemm_streamk<
  real,
  atomic_real2_ptr,
  M_group,
  N_group,
  K_group,
  M_group / 2,
  N_group / 2,
  K_simd
  >(A, B, C, C_block, gid.xy, sidx, lane_id);
}

kernel void hgemm_monolithic_batched
 (
  device void* A [[buffer(0)]],
  device void* B [[buffer(1)]],
  device half* C [[buffer(2)]],
  uint3 gid [[threadgroup_position_in_grid]],
  ushort sidx [[simdgroup_index_in_threadgroup]],
  ushort lane_id [[thread_index_in_simdgroup]])
{
  if (!is_function_constant_defined(B)) {
    return;
  }
  
  typedef half real;
  typedef atomic_half2_ptr atomic_real2_ptr;
  
  constexpr ushort M_group = 32; // threadgroup M
  constexpr ushort N_group = 32; // threadgroup N
  constexpr ushort K_group = 32; // threadgroup K
  
  constexpr ushort K_simd = 8; // simdgroup K
  constexpr ushort A_block_size = M_group * K_group;
  constexpr ushort B_block_size = K_group * N_group;
  
  threadgroup real C_block[A_block_size + B_block_size];
  
  // TODO: Support mixed precision, which will require not assuming A is a
  // buffer of floats or halfs.
  auto _A = (device half*)A + gid.z * A_stride;
  auto _B = (device half*)B + gid.z * B_stride;
  auto _C = (device half*)C + gid.z * C_stride;
  _gemm_streamk<
  real,
  atomic_real2_ptr,
  M_group,
  N_group,
  K_group,
  M_group / 2,
  N_group / 2,
  K_simd
  >(_A, _B, _C, C_block, gid.xy, sidx, lane_id);
}

#endif

kernel void sgemm
 (
  device void* A [[buffer(0)]],
  device void* B [[buffer(1)]],
  device float* C [[buffer(2)]],
#if STREAMK_PARALLEL_DECOMPOSITION
  device atomic_uint* locks [[buffer(3)]],
  device void* partialSums [[buffer(4)]],
  device uint* errorCodes [[buffer(5)]],
  
  // Instantiate 'g' CTAs.
  uint grid_size [[threadgroups_per_grid]],
  uint gid [[threadgroup_position_in_grid]],
#else
  uint2 gid [[threadgroup_position_in_grid]],
#endif
  ushort sidx [[simdgroup_index_in_threadgroup]],
  ushort lane_id [[thread_index_in_simdgroup]])
{
  typedef float real;
  typedef atomic_float2_ptr atomic_real2_ptr;
  
  constexpr ushort M_group = 32; // threadgroup M
  constexpr ushort N_group = 32; // threadgroup N
  constexpr ushort K_group = 32; // threadgroup K
  
  constexpr ushort K_simd = 8; // simdgroup K
  constexpr ushort A_block_size = M_group * K_group;
  constexpr ushort B_block_size = K_group * N_group;
  
  threadgroup real C_block[A_block_size + B_block_size];
  
  _gemm_streamk<
  real,
  atomic_real2_ptr,
  M_group,
  N_group,
  K_group,
  M_group / 2,
  N_group / 2,
  K_simd
  >(A, B, C, C_block, gid, sidx, lane_id);
}

kernel void sgemm_16x48
 (
  device void* A [[buffer(0)]],
  device void* B [[buffer(1)]],
  device float* C [[buffer(2)]],
#if STREAMK_PARALLEL_DECOMPOSITION
  device atomic_uint* locks [[buffer(3)]],
  device void* partialSums [[buffer(4)]],
  device uint* errorCodes [[buffer(5)]],
  
  // Instantiate 'g' CTAs.
  uint grid_size [[threadgroups_per_grid]],
  uint gid [[threadgroup_position_in_grid]],
#else
  uint2 gid [[threadgroup_position_in_grid]],
#endif
  ushort sidx [[simdgroup_index_in_threadgroup]],
  ushort lane_id [[thread_index_in_simdgroup]])
{
  typedef float real;
  typedef atomic_float2_ptr atomic_real2_ptr;
  
  constexpr ushort M_group = 16; // threadgroup M
  constexpr ushort N_group = 16; // threadgroup N
  constexpr ushort K_group = 48; // threadgroup K
  
  constexpr ushort K_simd = 8; // simdgroup K
  constexpr ushort A_block_size = M_group * K_group;
  constexpr ushort B_block_size = K_group * N_group;
  
  threadgroup real C_block[A_block_size + B_block_size];
  
  _gemm_streamk<
  real,
  atomic_real2_ptr,
  M_group,
  N_group,
  K_group,
  M_group / 2,
  N_group / 2,
  K_simd
  >(A, B, C, C_block, gid, sidx, lane_id);
}

kernel void sgemm_16x48_batched
 (
  device void* A [[buffer(0)]],
  device void* B [[buffer(1)]],
  device float* C [[buffer(2)]],
  constant ulong4 *ABC_offsets [[buffer(3)]],
#if STREAMK_PARALLEL_DECOMPOSITION
  device atomic_uint* locks [[buffer(3)]],
  device void* partialSums [[buffer(4)]],
  device uint* errorCodes [[buffer(5)]],
  
  // Instantiate 'g' CTAs.
  uint grid_size [[threadgroups_per_grid]],
  uint gid [[threadgroup_position_in_grid]],
#else
  uint3 gid [[threadgroup_position_in_grid]],
#endif
  ushort sidx [[simdgroup_index_in_threadgroup]],
  ushort lane_id [[thread_index_in_simdgroup]])
{
  typedef float real;
  typedef atomic_float2_ptr atomic_real2_ptr;
  
  constexpr ushort M_group = 16; // threadgroup M
  constexpr ushort N_group = 16; // threadgroup N
  constexpr ushort K_group = 48; // threadgroup K
  
  constexpr ushort K_simd = 8; // simdgroup K
  constexpr ushort A_block_size = M_group * K_group;
  constexpr ushort B_block_size = K_group * N_group;
  
  threadgroup real C_block[A_block_size + B_block_size];
  
  ulong3 offsets = ABC_offsets[gid.z].xyz;
  auto _A = (device float*)A + offsets[0];
  auto _B = (device float*)B + offsets[1];
  auto _C = (device float*)C + offsets[2];
  
  _gemm_streamk<
  real,
  atomic_real2_ptr,
  M_group,
  N_group,
  K_group,
  M_group / 2,
  N_group / 2,
  K_simd
  >(_A, _B, _C, C_block, gid.xy, sidx, lane_id);
}

kernel void sgemm_32x32
 (
  device void* A [[buffer(0)]],
  device void* B [[buffer(1)]],
  device float* C [[buffer(2)]],
#if STREAMK_PARALLEL_DECOMPOSITION
  device atomic_uint* locks [[buffer(3)]],
  device void* partialSums [[buffer(4)]],
  device uint* errorCodes [[buffer(5)]],
  
  // Instantiate 'g' CTAs.
  uint grid_size [[threadgroups_per_grid]],
  uint gid [[threadgroup_position_in_grid]],
#else
  uint2 gid [[threadgroup_position_in_grid]],
#endif
  ushort sidx [[simdgroup_index_in_threadgroup]],
  ushort lane_id [[thread_index_in_simdgroup]])
{
  typedef float real;
  typedef atomic_float2_ptr atomic_real2_ptr;
  
  constexpr ushort M_group = 32; // threadgroup M
  constexpr ushort N_group = 32; // threadgroup N
  constexpr ushort K_group = 32; // threadgroup K
  
  constexpr ushort K_simd = 8; // simdgroup K
  constexpr ushort A_block_size = M_group * K_group;
  constexpr ushort B_block_size = K_group * N_group;
  
  threadgroup real C_block[A_block_size + B_block_size];
  
  _gemm_streamk<
  real,
  atomic_real2_ptr,
  M_group,
  N_group,
  K_group,
  M_group / 2,
  N_group / 2,
  K_simd
  >(A, B, C, C_block, gid, sidx, lane_id);
}

kernel void sgemm_32x32_batched
 (
  device void* A [[buffer(0)]],
  device void* B [[buffer(1)]],
  device float* C [[buffer(2)]],
  constant ulong4 *ABC_offsets [[buffer(3)]],
#if STREAMK_PARALLEL_DECOMPOSITION
  device atomic_uint* locks [[buffer(3)]],
  device void* partialSums [[buffer(4)]],
  device uint* errorCodes [[buffer(5)]],
  
  // Instantiate 'g' CTAs.
  uint grid_size [[threadgroups_per_grid]],
  uint gid [[threadgroup_position_in_grid]],
#else
  uint3 gid [[threadgroup_position_in_grid]],
#endif
  ushort sidx [[simdgroup_index_in_threadgroup]],
  ushort lane_id [[thread_index_in_simdgroup]])
{
  typedef float real;
  typedef atomic_float2_ptr atomic_real2_ptr;
  
  constexpr ushort M_group = 32; // threadgroup M
  constexpr ushort N_group = 32; // threadgroup N
  constexpr ushort K_group = 32; // threadgroup K
  
  constexpr ushort K_simd = 8; // simdgroup K
  constexpr ushort A_block_size = M_group * K_group;
  constexpr ushort B_block_size = K_group * N_group;
  
  threadgroup real C_block[A_block_size + B_block_size];
  
  ulong3 offsets = ABC_offsets[gid.z].xyz;
  auto _A = (device float*)A + offsets[0];
  auto _B = (device float*)B + offsets[1];
  auto _C = (device float*)C + offsets[2];
  
  _gemm_streamk<
  real,
  atomic_real2_ptr,
  M_group,
  N_group,
  K_group,
  M_group / 2,
  N_group / 2,
  K_simd
  >(_A, _B, _C, C_block, gid.xy, sidx, lane_id);
}

kernel void sgemm_48x24
 (
  device void* A [[buffer(0)]],
  device void* B [[buffer(1)]],
  device float* C [[buffer(2)]],
#if STREAMK_PARALLEL_DECOMPOSITION
  device atomic_uint* locks [[buffer(3)]],
  device void* partialSums [[buffer(4)]],
  device uint* errorCodes [[buffer(5)]],
  
  // Instantiate 'g' CTAs.
  uint grid_size [[threadgroups_per_grid]],
  uint gid [[threadgroup_position_in_grid]],
#else
  uint2 gid [[threadgroup_position_in_grid]],
#endif
  ushort sidx [[simdgroup_index_in_threadgroup]],
  ushort lane_id [[thread_index_in_simdgroup]])
{
  typedef float real;
  typedef atomic_float2_ptr atomic_real2_ptr;
  
  constexpr ushort M_group = 48; // threadgroup M
  constexpr ushort N_group = 48; // threadgroup N
  constexpr ushort K_group = 24; // threadgroup K
  
  constexpr ushort K_simd = 8; // simdgroup K
  constexpr ushort A_block_size = M_group * K_group;
  constexpr ushort B_block_size = K_group * N_group;
  
  threadgroup real C_block[A_block_size + B_block_size];
  
  _gemm_streamk<
  real,
  atomic_real2_ptr,
  M_group,
  N_group,
  K_group,
  M_group / 2,
  N_group / 2,
  K_simd
  >(A, B, C, C_block, gid, sidx, lane_id);
}

kernel void sgemm_48x24_batched
 (
  device void* A [[buffer(0)]],
  device void* B [[buffer(1)]],
  device float* C [[buffer(2)]],
  constant ulong4 *ABC_offsets [[buffer(3)]],
#if STREAMK_PARALLEL_DECOMPOSITION
  device atomic_uint* locks [[buffer(3)]],
  device void* partialSums [[buffer(4)]],
  device uint* errorCodes [[buffer(5)]],
  
  // Instantiate 'g' CTAs.
  uint grid_size [[threadgroups_per_grid]],
  uint gid [[threadgroup_position_in_grid]],
#else
  uint3 gid [[threadgroup_position_in_grid]],
#endif
  ushort sidx [[simdgroup_index_in_threadgroup]],
  ushort lane_id [[thread_index_in_simdgroup]])
{
  typedef float real;
  typedef atomic_float2_ptr atomic_real2_ptr;
  
  constexpr ushort M_group = 48; // threadgroup M
  constexpr ushort N_group = 48; // threadgroup N
  constexpr ushort K_group = 24; // threadgroup K
  
  constexpr ushort K_simd = 8; // simdgroup K
  constexpr ushort A_block_size = M_group * K_group;
  constexpr ushort B_block_size = K_group * N_group;
  
  threadgroup real C_block[A_block_size + B_block_size];
  
  ulong3 offsets = ABC_offsets[gid.z].xyz;
  auto _A = (device float*)A + offsets[0];
  auto _B = (device float*)B + offsets[1];
  auto _C = (device float*)C + offsets[2];
  
  _gemm_streamk<
  real,
  atomic_real2_ptr,
  M_group,
  N_group,
  K_group,
  M_group / 2,
  N_group / 2,
  K_simd
  >(_A, _B, _C, C_block, gid.xy, sidx, lane_id);
}

//kernel void sgemm_batched
// (
//  device void* A [[buffer(0)]],
//  device void* B [[buffer(1)]],
//  device float* C [[buffer(2)]],
//  
//  // A, B, and C offsets
//  
//  uint3 gid [[threadgroup_position_in_grid]],
//  ushort sidx [[simdgroup_index_in_threadgroup]],
//  ushort lane_id [[thread_index_in_simdgroup]])
//{
//  typedef float real;
//  typedef atomic_float2_ptr atomic_real2_ptr;
//  
//  constexpr ushort M_group = 32; // threadgroup M
//  constexpr ushort N_group = 32; // threadgroup N
//  constexpr ushort K_group = 32; // threadgroup K
//  
//  constexpr ushort K_simd = 8; // simdgroup K
//  constexpr ushort A_block_size = M_group * K_group;
//  constexpr ushort B_block_size = K_group * N_group;
//  
//  threadgroup real C_block[A_block_size + B_block_size];
//  
//  _gemm_streamk<
//  real,
//  atomic_real2_ptr,
//  M_group,
//  N_group,
//  K_group,
//  M_group / 2,
//  N_group / 2,
//  K_simd
//  >(A, B, C, C_block, gid, sidx, lane_id);
//}

// sgemm_batched

#ifdef MFA_MONOLITHIC_KERNELS

// Part of the public API, but doesn't require block size to be set through
// function constants.
kernel void sgemm_monolithic
 (
  device void* A [[buffer(0)]],
  device void* B [[buffer(1)]],
  device float* C [[buffer(2)]],
  uint2 gid [[threadgroup_position_in_grid]],
  ushort sidx [[simdgroup_index_in_threadgroup]],
  ushort lane_id [[thread_index_in_simdgroup]])
{
  typedef float real;
  typedef atomic_float2_ptr atomic_real2_ptr;
  
  constexpr ushort M_group = 32; // threadgroup M
  constexpr ushort N_group = 32; // threadgroup N
  constexpr ushort K_group = 32; // threadgroup K
  
  constexpr ushort K_simd = 8; // simdgroup K
  constexpr ushort A_block_size = M_group * K_group;
  constexpr ushort B_block_size = K_group * N_group;
  
  threadgroup real C_block[A_block_size + B_block_size];
  
  _gemm_streamk<
  real,
  atomic_real2_ptr,
  M_group,
  N_group,
  K_group,
  M_group / 2,
  N_group / 2,
  K_simd
  >(A, B, C, C_block, gid.xy, sidx, lane_id);
}

kernel void sgemm_monolithic_batched
 (
  device void* A [[buffer(0)]],
  device void* B [[buffer(1)]],
  device float* C [[buffer(2)]],
  uint3 gid [[threadgroup_position_in_grid]],
  ushort sidx [[simdgroup_index_in_threadgroup]],
  ushort lane_id [[thread_index_in_simdgroup]])
{
  if (!is_function_constant_defined(B)) {
    return;
  }
  
  typedef float real;
  typedef atomic_float2_ptr atomic_real2_ptr;
  
  constexpr ushort M_group = 32; // threadgroup M
  constexpr ushort N_group = 32; // threadgroup N
  constexpr ushort K_group = 32; // threadgroup K
  
  constexpr ushort K_simd = 8; // simdgroup K
  constexpr ushort A_block_size = M_group * K_group;
  constexpr ushort B_block_size = K_group * N_group;
  
  threadgroup real C_block[A_block_size + B_block_size];
  
  // TODO: Support mixed precision, which will require not assuming A is a
  // buffer of floats or halfs.
  auto _A = (device float*)A + gid.z * A_stride;
  auto _B = (device float*)B + gid.z * B_stride;
  auto _C = (device float*)C + gid.z * C_stride;
  _gemm_streamk<
  real,
  atomic_real2_ptr,
  M_group,
  N_group,
  K_group,
  M_group / 2,
  N_group / 2,
  K_simd
  >(_A, _B, _C, C_block, gid.xy, sidx, lane_id);
}

#endif

// TODO: hgemm_batched, sgemm_batched

// TODO: hgemm_dynamic, sgemm_dynamic
// All parameters defined at runtime, accepts 3D* thread indices, optimized to
// minimize compile time or permit dynamic shapes. This would be necessary for
// the key-value cache in LLaMA.
//
// *If ICBs cause a fatal detriment to GPU-side sequential throughput:
//
// After querying the optimal parameters for each matrix separately, you should
// write them to a buffer. Then, dispatch a number of threadgroups equalling the
// sum of each matrix in the batch. In case the matrices are all the same,
// there is an argument for specifying an index within the batch of parameters.
// Each matrix can recycle the same parameters or use different ones.
//
// *If not:
//
// To perform a dynamic batched multiplication, you must encode each matmul
// in a separate command using ICBs. The GPU can perform multiple commands
// simultaneously, equivalent to how many actual (not virtual) threadgroups are
// present in the core (3-4).

#undef MN_MACC_ELISIONS
#undef MN_LOAD_ELISIONS
#undef C_DIRECT_STORE

#undef STREAMK_PARALLEL_DECOMPOSITION
#undef STREAMK_FAULT_COUNTERS
#undef FAULT_COUNTER_INIT
#undef FAULT_COUNTER_INCREMENT

#undef MAKE_GEMM_STREAMK

#endif // #if defined(__HAVE_SIMDGROUP_FUTURE__)

#endif // __METAL__
