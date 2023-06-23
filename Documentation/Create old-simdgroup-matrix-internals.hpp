//
//  simdgroup_matrix_internals.hpp
//  MetalFlashAttention
//
//  Created by Philip Turner on 6/16/23.
//

#ifndef simdgroup_matrix_internals_hpp
#define simdgroup_matrix_internals_hpp

#ifdef __METAL__
#include <metal_stdlib>
using namespace metal;

// Hacking SIMD-group matrix:
// https://patents.google.com/patent/US11256518B2
//
// Convention for indexing elements:
//
// 00 01 02 03 04 05 06 07
// 10 11 12 13 14 15 16 17
// 20 21 22 23 24 25 26 27
// ...
//
// Storage format among threads:
//
// [00 01] [02 03] [10 11] [12 13] <- Quad 0
// [20 21] [22 23] [30 31] [32 33] <- Quad 1
// [04 05] [06 07] [14 15] [16 17] <- Quad 2
// [24 25] [26 27] [34 35] [36 37] <- Quad 3
// [40 41] [42 43] [50 51] [52 53] <- Quad 4
// [60 61] [62 63] [70 71] [72 73] <- Quad 5
// ...
//
// Matrix subdivision across quads:
//
// Quad 0 Quad 2
// Quad 1 Quad 3
// Quad 4 Quad 6
// Quad 5 Quad 7
//
// Matrix subdivision within a quad:
//
// Thread 0 Thread 1
// Thread 2 Thread 3
//
// Element assignment across threads:
//
//  0  0  1  1 |  8  8  9  9
//  2  2  3  3 | 10 10 11 11
//  4  4  5  5 | 12 12 13 13
//  6  6  7  7 | 14 14 15 15
// ----------- | -----------
// 16 16 17 17 | 24 24 25 25
// 18 18 19 19 | 26 26 27 27
// 20 20 21 21 | 28 28 29 29
// 22 22 23 23 | 30 30 31 31

// 'quad' != 'quadrant'
// 'quad' = group of 4 threads
// 'quadrant' = 1/4 of the SIMD matrix

__attribute__((__always_inline__)) inline
ushort get_simdgroup_matrix_m(ushort lane_id, ushort quad_id) {
  // Compute the thread's M index.
  constexpr ushort QUADRANT_SPAN_M = 4;
  constexpr ushort THREADS_PER_QUADRANT = 8;
  ushort M_floor_of_quadrant = (quad_id / 4) * QUADRANT_SPAN_M;
  ushort M_in_quadrant = (lane_id / 2) % (THREADS_PER_QUADRANT / 2);
  return M_floor_of_quadrant + M_in_quadrant;
}

__attribute__((__always_inline__)) inline
ushort get_simdgroup_matrix_n(ushort lane_id, ushort quad_id) {
  // Compute the thread's N index.
  ushort N_floor_of_quadrant = (quad_id & 2) * 2; // 0 or 4
  ushort N_in_quadrant = (lane_id % 2) * 2; // 0 or 2
  return N_floor_of_quadrant + N_in_quadrant;
}

// TODO: When indexing the device pointer, explicitly cast the product of `m`
// and `N` to `ulong` for very large matrices. This should be a conditional
// compile path that's disabled when you can guarantee the matrix is smaller
// than 4 billion elements. For now, such large of matrices are not supported.

// TODO: The atomic memory accesses are not coalesced, so they might
// underutilize bandwidth by 50%. We may need to rearrange the data in registers
// before/after accessing the partial sums for Float32.

template <typename real, typename real_ptr, typename real2_ptr, typename index>
inline void vec_load2
(
 thread simdgroup_matrix<real, 8, 8>& value,
 const real_ptr block, index tile_width, vec<ushort, 2> origin)
{
  auto src = (real2_ptr)(block + origin.y * tile_width + origin.x);
  reinterpret_cast<thread vec<real, 2>&>(value) = *src;
}

template <typename real, typename real_ptr, typename real2_ptr, typename index>
inline void vec_load2_transpose
(
 thread simdgroup_matrix<real, 8, 8>& value,
 const real_ptr block, index tile_width, vec<ushort, 2> origin)
{
  auto src0 = (real_ptr)(block + origin.x * tile_width + origin.y);
  auto src1 = src0 + tile_width;
  reinterpret_cast<thread vec<real, 2>&>(value) = { *src0, *src1 };
}

template <typename real, typename real_ptr, typename real2_ptr, typename index>
inline void vec_accumulate2
(
 thread simdgroup_matrix<real, 8, 8>& value,
 const real_ptr block, index tile_width, vec<ushort, 2> origin)
{
  auto src = (real2_ptr)(block + origin.y * tile_width + origin.x);
  reinterpret_cast<thread vec<real, 2>&>(value) += *src;
}

template <typename real, typename real_ptr, typename real2_ptr, typename index>
inline void vec_store2
(
 const simdgroup_matrix<real, 8, 8> value,
 real_ptr block, index tile_width, vec<ushort, 2> origin)
{
  auto dst = (real2_ptr)(block + origin.y * tile_width + origin.x);
  *dst = reinterpret_cast<const thread vec<real, 2>&>(value);
}

template <typename real, typename real_ptr, typename real2_ptr, typename index>
inline void vec_store1
(
 const simdgroup_matrix<real, 8, 8> value,
 real_ptr block, index tile_width, vec<ushort, 2> origin)
{
  auto dst = (real_ptr)(block + origin.y * tile_width + origin.x);
  *dst = reinterpret_cast<const thread real&>(value);
}

namespace {
  struct atomic_half2_ptr {
    // Must be aligned to a multiple of 2 elements.
    device half* ptr;
    
    // Tile width must be a 16-bit integer.
    bool atomic_compare
    (
     thread simdgroup_matrix<half, 8, 8>& value,
     ushort tile_width, vec<ushort, 2> origin)
    {
      auto src = (device atomic_uint*)(ptr + origin.y * tile_width + origin.x);
      auto _src = atomic_load_explicit(src, memory_order_relaxed);
      auto _actual = as_type<half2>(_src);
      auto _value = reinterpret_cast<const thread half2&>(value);
      return (_actual[0] == _value[0]) && (_actual[1] == _value[1]);
    }
    
    // Tile width must be a 16-bit integer.
    void atomic_accumulate
    (
     thread simdgroup_matrix<half, 8, 8>& value,
     ushort tile_width, vec<ushort, 2> origin)
    {
      auto src = (device atomic_uint*)(ptr + origin.y * tile_width + origin.x);
      auto _src = atomic_load_explicit(src, memory_order_relaxed);
      auto _actual = as_type<half2>(_src);
      reinterpret_cast<thread half2&>(value)[0] += _actual[0];
      reinterpret_cast<thread half2&>(value)[1] += _actual[1];
    }
    
    // Tile width must be a 16-bit integer.
    void atomic_store
    (
     thread simdgroup_matrix<half, 8, 8>& value,
     ushort tile_width, vec<ushort, 2> origin)
    {
      auto dst = (device atomic_uint*)(ptr + origin.y * tile_width + origin.x);
      auto _value = reinterpret_cast<const thread half2&>(value);
      uint _dst = as_type<uint>(_value);
      atomic_store_explicit(dst, _dst, memory_order_relaxed);
    }
  };
  
  struct atomic_float2_ptr {
    // Must be aligned to a multiple of 2 elements.
    device float* ptr;
    
    // Tile width must be a 16-bit integer.
    bool atomic_compare
    (
     thread simdgroup_matrix<float, 8, 8>& value,
     ushort tile_width, vec<ushort, 2> origin)
    {
      auto src = (device atomic_float*)(ptr + origin.y * tile_width + origin.x);
      auto _value = reinterpret_cast<const thread float2&>(value);
      auto _actual0 = atomic_load_explicit(src + 0, memory_order_relaxed);
      auto _actual1 = atomic_load_explicit(src + 1, memory_order_relaxed);
      return (_actual0 == _value[0]) && (_actual1 == _value[1]);
    }
    
    // Tile width must be a 16-bit integer.
    void atomic_accumulate
    (
     thread simdgroup_matrix<float, 8, 8>& value,
     ushort tile_width, vec<ushort, 2> origin)
    {
      auto src = (device atomic_float*)(ptr + origin.y * tile_width + origin.x);
      auto _src0 = atomic_load_explicit(src, memory_order_relaxed);
      auto _src1 = atomic_load_explicit(src + 1, memory_order_relaxed);
      reinterpret_cast<thread float2&>(value)[0] += _src0;
      reinterpret_cast<thread float2&>(value)[1] += _src1;
    }
    
    // Tile width must be a 16-bit integer.
    void atomic_store
    (
     const simdgroup_matrix<float, 8, 8> value,
     ushort tile_width, vec<ushort, 2> origin)
    {
      auto src = (device atomic_float*)(ptr + origin.y * tile_width + origin.x);
      auto _value = reinterpret_cast<const thread float2&>(value);
      atomic_store_explicit(src + 0, _value[0], memory_order_relaxed);
      atomic_store_explicit(src + 1, _value[1], memory_order_relaxed);
    }
  };
}
#endif

#endif /* simdgroup_matrix_internals_hpp */
