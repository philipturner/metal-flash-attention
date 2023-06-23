//
//  FlashAttention.metal
//  MetalFlashAttention
//
//  Created by Philip Turner on 5/23/23.
//

#include <metal_stdlib>
#include "../Common/Common.hpp"
#include "../Common/SIMDFuturesPlaceholder.hpp"
#include "../Common/simdgroup_matrix_internals.hpp"
using namespace metal;

#if 1

// WARNING:
//
// This kernel is 100% untested. It has never actually been run, and will
// probably fail. It is a draft that needs to be rewritten from scratch.

// MARK: - GPU Core SRAM

// This is for Apple 7-8. Not sure about Apple 9, which might let simds access
// much larger amounts of `thread` SRAM than possible before.
//
// SRAM type     | per core | max/threadgroup |
// ------------- | -------- | --------------- |
// `thread`      | 384 KB   | 192 KB          |
// `threadgroup` |  64 KB   |  32 KB          |
//
// MARK: - SRAM Allocation Sizes
//
//          max `thread` allocation | max `threadgroup` allocation
//                                  |
// |               |   per |    per |            simds/group            |
// |     occupancy |  simd |  group |      4 |      8 |     12 |     24 |
// | ------------- | ----- | ------ | ------ | ------ | ------ | ------ |
// | 96 simds/core |  4 KB |  48 KB | 2.5 KB |   5 KB |   8 KB |  16 KB |
// | 48 simds/core |  8 KB |  96 KB |   5 KB |  10 KB |  16 KB |  32 KB |
// | 24 simds/core | 16 KB | 192 KB |  10 KB |  20 KB |  32 KB |        |
// | 16 simds/core | 16 KB | 128 KB |  16 KB |  32 KB |        |        |
// |  8 simds/core | 16 KB |  64 KB |  32 KB |        |        |        |
//                                  |
// Converting KB into SIMD matrices (FP16). Matrices for FP32 is half this.
//                                  |
// |               |   per |    per |            simds/group            |
// |     occupancy |  simd |  group |      4 |      8 |     12 |     24 |
// | ------------- | ----- | ------ | ------ | ------ | ------ | ------ |
// | 96 simds/core |    32 |    384 |     20 |     40 |   8 KB |  16 KB |
// | 48 simds/core |    64 |    768 |     40 |     80 |  16 KB |  32 KB |
// | 24 simds/core |   128 |   1536 |     80 |    160 |  32 KB |        |
// | 16 simds/core |   128 |   1024 |    128 |    256 |        |        |
// |  8 simds/core |   128 |    512 |    256 |        |        |        |

// MARK: - Block Sizes

// We pre-compile several variants with fixed allocations for
// number of heads. ALU utilization scales inversely with number of heads, and
// the shader doesn't support more than 256 heads. However, ALU utilization
// maxes out at D = 32-48 (40-64 simds/core) for long sequence lengths.
//
// Exemplar block sizes:
// - 4x1, Br=64, Br=32: best suited for large sequence lengths
//   - simds/GPU = 4 * N/64
// - 1x8, Br=8, Bc=128: best suited for small sequence lengths
//   - simds/GPU = 4 * N/8
//
// TODO: Examine situations with non-power-2 splits along the R dimension.
//
// simds/group | 4x1 | 2x2 | 3x4 |  1x4 |
// R_group |      64 |  32 |  24 |    8 |
// R_simd  |      16 |  16 |   8 |    8 |
// ------- | ------- | --- | --- | ---- |
// C_group |      32 |  32 |  64 |  128 |
// C_simd  |      32 |  16 |  16 |   32 |
//
//                                simds/core
//
// The first row seems to actually be Br=64, Bc=64.
//
//    D  | F16 | F32 |
// | --- | --------- |  TODO: Finish the table after finding good block sizes
// | 256 |   8 |   ~ |  empirically.
// | 192 |  10 |   ~ |
// | 160 |  12 |   ~ |
// | 128 |  16 |   8 |
// |  96 |  20 |  10 |
// |  80 |  24 |  12 |
// |  64 |  32 |  16 |
// |  48 |  40 |  20 |
// |  40 |  48 |  24 |
// |  32 |  64 |  32 |
// |  24 |  80 |  40 |
// |  16 |  96 |  48 |
// |   8 |  96 |  96 |

constant uint R [[function_constant(100)]];
constant uint C [[function_constant(101)]];
constant ushort D [[function_constant(9000)]]; // also set 102

// Whether each matrix is transposed in `device` RAM. If K is not transposed or
// the function constant `K_trans` is not set, the shader will transpose K when
// reading from `threadgroup` SRAM to registers. This may harm performance.
// WARNING: Not recognized yet.
constant bool Q_trans [[function_constant(103)]];
constant bool K_trans [[function_constant(104)]];
constant bool V_trans [[function_constant(105)]];
constant bool O_trans [[function_constant(106)]];

// Alpha and beta constants from BLAS. These might be used, for example, to
// divide by sqrt(D) during scaled dot-product attention. In that case, set
// `QK_T_alpha` to 1 / sqrt(float(D)).
constant float QK_T_alpha [[function_constant(107)]];
// Index 108 reserved for QK_T_beta, which cannot exist because the NxN
// intermediate attention matrix is transient.
// Indices 109-110 reserved for O_alpha and O_beta.

constant ulong Q_stride [[function_constant(111)]];
constant ulong K_stride [[function_constant(112)]];
constant ulong V_stride [[function_constant(113)]];
constant ulong O_stride [[function_constant(114)]];

// For now, the number of concurrent dispatches during batched attention must be
// known at compile-time.
constant uint B [[function_constant(115)]];

// One of the codes from `metal_flash_attention::data_type`.
// Currently, all four precisions must be the same.
constant ushort Q_precision [[function_constant(120)]];
constant ushort K_precision [[function_constant(121)]];
constant ushort V_precision [[function_constant(122)]];
constant ushort O_precision [[function_constant(123)]];
// This should be enforced through a constant-folding expression that makes the
// function return early.

// TODO: Optional fusing of dense layers for queries and outputs.

// MARK: - Outputs of `configure_attention`

// Good starting config:
// D_simd = 64
// R_simd = 16
// C_simd = 32
// These must all be multiples of 8.
constant ushort R_simd [[function_constant(200)]];
constant ushort C_simd [[function_constant(201)]];
// Index 202 reserved for `D_simd`.

// C_splits must be a power of 2.
constant ushort R_splits [[function_constant(210)]];
constant ushort C_splits [[function_constant(211)]];
// Index 212 reserved for `D_splits`.

// R_group must not exceed C_group.
constant ushort R_group = R_simd * R_splits;
constant ushort C_group = C_simd * C_splits;
constant ushort D_edge = (D % 8 == 0) ? D : (~7 & (D + 7));

// Although not a function constant, `threadgroup_block_bytes` should be the
// next output of `configure_gemm`. We reserve indices 220 - 229 for this.
//
// The threadgroup memory block must be this large:
// D_edge * C_group * sizeof(real)
constant ushort threadgroup_block_elements = D_edge * C_group;
constant ushort Q_offset = 0;

// Recycle Q to store the other simd's L/M when combining partials.
constant ushort SP_max_cols = (C_splits == 1) ? 8 : 16;
constant ushort O_offset = Q_offset + R_simd * max(SP_max_cols, D_edge);

// To allow the accumulator to be FP32, double the number of matrices.
constant bool O_upcast = (O_precision == metal_flash_attention::HALF);
constant ushort SP_offset = O_offset + (O_upcast ? 2 : 1) * R_simd * D_edge;
constant ushort KV_offset = SP_offset + R_simd * C_simd;
constant ushort K_span = 8 * C_simd;
constant ushort V_span = 8 * D_edge;

// For FP16, we always consume two halfs (one float), which is one SIMD matrix
// per row of SIMD matrices in P. For FP32, we waste an extra 32-bit register
// for now.
constant ushort L_offset = KV_offset + max(K_span, V_span);
constant ushort M_offset = L_offset + R_simd * 8;
constant ushort thread_block_elements = M_offset + R_simd * 8;
constant ushort thread_block_matrices = thread_block_elements / 64;

// Recompute this to save registers, but don't hard-code the heuristic.
inline uint get_i(uint2 gid) {
  return gid.x * R_group;
}

// Instead of computing exp(x - max), we compute exp2(x * log_2(e) -
// max * log_2(e)) This allows the compiler to use the ffma instruction
// instead of fadd and fmul separately. Source:
// https://github.com/HazyResearch/flash-attention/blob/9818f85fee29ac6b60c9214bce841f8109a18b1b/csrc/flash_attn/src/fmha/softmax.h#L222-L224
//
// The `sub` operand must be `m_new * M_LOG2E_F`, but not negated yet. This
// includes the `mult` parameter so you can fuse the scaling by `QK_T_alpha`
// with zero cost at runtime.
template <typename real>
inline float quick_exp(real lhs, float mult, float sub) {
  return fast::exp2(fast::fma(real(lhs), float(mult), float(-sub)));
}
inline float _S_mult() {
  if (is_function_constant_defined(QK_T_alpha)) {
    return M_LOG2E_F * QK_T_alpha;
  } else {
    return M_LOG2E_F;
  }
}
constant float S_mult = _S_mult();

// A chunk `thread` SRAM, which must be allocated at compile-time. Chunks of
// `threadgroup` SRAM are allocated at encode-time.
//
// Granularity: blocks of 64 elements
// - FP16: 128 bytes
// - FP32: 256 bytes
//
// Has utilities to help with emulating multiple different block shapes, using a
// single `thread` SRAM allocation. It returns the index, in SIMD matrices, from
// the start of the allocation.
//
// If `d_or_c` never reaches the maximum possible value, the Metal compiler
// should optimize away the registers. In that case, a single set of AIR code
// could support all possible permutations. In either case, the source code will
// look similar.
template <typename real, ushort matrices>
struct thread_allocation {
  typedef vec<real, 2> real2;
  typedef simdgroup_matrix<real, 8, 8> simdgroup_real8x8;
  
  thread simdgroup_real8x8 sram[matrices];
  ushort R_in_simd;
  ushort C_in_simd;
  
  METAL_FUNC
  thread_allocation(ushort R_in_simd, ushort C_in_simd) {
    this->R_in_simd = R_in_simd;
    this->C_in_simd = C_in_simd;
  }
  
private:
  METAL_FUNC
  thread simdgroup_real8x8* _index(ushort offset, ushort columns,
                                   ushort row, ushort column) {
    return sram + (offset + row * columns) / 64 + column / 8;
  }
  
  METAL_FUNC
  thread simdgroup_float8x8* _index32(ushort offset, ushort columns,
                                      ushort row, ushort column) {
    auto out = (thread simdgroup_float8x8*)(sram + offset / 64);
    return out + (row * columns) / 64 + column / 8;
  }
  
public:
  METAL_FUNC thread simdgroup_real8x8* q(ushort r, ushort d) {
    return _index(Q_offset, D_edge, r, d);
  }
  
  METAL_FUNC thread simdgroup_float8x8* o(ushort r, ushort d) {
    return _index32(O_offset, D_edge, r, d);
  }
  
  METAL_FUNC thread simdgroup_real8x8* s(ushort r, ushort c) {
    return _index(SP_offset, C_simd, r, c);
  }
  
  METAL_FUNC thread simdgroup_real8x8* p(ushort r, ushort c) {
    return _index(SP_offset, C_simd, r, c);
  }
  
  METAL_FUNC thread simdgroup_real8x8* k_transpose(ushort d, ushort c) {
    return _index(KV_offset, C_simd, d, c);
  }
  
  METAL_FUNC thread simdgroup_real8x8* v(ushort c, ushort d) {
    return _index(KV_offset, D_edge, c, d);
  }
  
  METAL_FUNC thread float* l(ushort r) {
    return reinterpret_cast<thread float*>(_index(L_offset, 8, r, 0));
  }
  
  METAL_FUNC thread float* m(ushort r) {
    return reinterpret_cast<thread float*>(_index(M_offset, 8, r, 0));
  }
  
  METAL_FUNC thread float* l_other(ushort r) {
    return reinterpret_cast<thread float*>(_index(O_offset, 8, r, 0));
  }
  
  METAL_FUNC thread float* m_other(ushort r) {
    return reinterpret_cast<thread float*>(_index(O_offset, 8, r, 8));
  }
  
  template <typename T>
  METAL_FUNC
  vec<T, 2> get(thread simdgroup_matrix<T, 8, 8>* matrix) {
    auto thread_elements = (*matrix).thread_elements();
    return vec<T, 2>(thread_elements[0], thread_elements[1]);
  }
  
  template <typename T>
  METAL_FUNC
  void set(thread simdgroup_matrix<T, 8, 8>* matrix, vec<T, 2> data) {
    (*matrix).thread_elements()[0] = data[0];
    (*matrix).thread_elements()[1] = data[1];
  }
  
  template <typename T>
  METAL_FUNC void apply_mask(thread vec<T, 2>* data, real mask, short trail) {
    if (C % 2 == 0) {
      if (trail < 0) {
        data[0] = mask;
        data[1] = mask;
      }
    } else {
      if (trail < 0) {
        data[0] = mask;
      }
      if (trail < 1) {
        data[1] = mask;
      }
    }
  }
  
  METAL_FUNC
  void reduce_max(ushort r, ushort2 bounds, thread real *out) {
#pragma clang loop unroll(full)
    for (ushort c = bounds[0]; c < bounds[1]; c += 8) {
      // S and P are the same allocation; it doesn't matter which one we choose.
      auto S = this->get(this->s(r, c));
      
      // Compute the partial while leveraging instruction-level parallelism.
      real partial = max(S[0], S[1]);
      if (c == 0) {
        *out = partial;
      } else {
        *out = max(*out, partial);
      }
    }
  }
  
  METAL_FUNC
  void reduce_sum(ushort r, ushort2 bounds, thread float *out, float sub) {
#pragma clang loop unroll(full)
    for (ushort c = bounds[0]; c < bounds[1]; c += 8) {
      // S and P are the same allocation; it doesn't matter which one we choose.
      auto S = this->get(this->s(r, c));
      float2 P = {
        quick_exp(S[0], S_mult, sub),
        quick_exp(S[1], S_mult, sub)
      };
      this->set(this->s(r, c), real2(P));
      
      float partial = P[0] + P[1];
      if (c == 0) {
        *out = partial;
      } else {
        *out += partial;
      }
    }
  }
  
  METAL_FUNC
  void load(ushort offset, threadgroup real* src, ushort2 tile) {
    auto _src = src + R_in_simd * tile.x + C_in_simd;
#pragma clang loop unroll(full)
    for (ushort row = 0; row < tile.y; row += 8) {
#pragma clang loop unroll(full)
      for (ushort col = 0; col < tile.x; col += 8) {
        auto matrix = _index(offset, tile.x, row, col);
        vec_load2<
        real, threadgroup real*, threadgroup real2*, ushort
        >(*matrix, _src, tile.x, ushort2(col, row));
      }
    }
  }
  
  METAL_FUNC
  void load_transpose(ushort offset, threadgroup real* src, ushort2 tile) {
    // Switch the order of `C_in_simd` and `R_in_simd` for transpose.
    auto _src = src + C_in_simd * tile.x + R_in_simd;
#pragma clang loop unroll(full)
    for (ushort row = 0; row < tile.y; row += 8) {
#pragma clang loop unroll(full)
      for (ushort col = 0; col < tile.x; col += 8) {
        auto matrix = _index(offset, tile.y, col, row);
        vec_load2_transpose<
        real, threadgroup real*, threadgroup real2*, ushort
        >(*matrix, _src, tile.x, ushort2(col, row));
      }
    }
  }
  
  // Before storing, you must divide O by `l_i` and cast from FP32 to the
  // original precision. The result is stored in Q's (now unused) allocation.
  METAL_FUNC
  void prepare_o(bool is_first_half, bool maybe_skip_divide) {
    if (maybe_skip_divide && !O_upcast) {
      return;
    }
    const ushort R_half = ~7 & (R_simd / 2 + 7);
    const ushort start = (is_first_half ? 0 : R_half);
    const ushort end = (is_first_half ? R_half : R_simd);
    
#pragma clang loop unroll(full)
    for (ushort row = start; row < end / 2; row += 8) {
      float l_inv = fast::divide(1, *(this->l(row)));
#pragma clang loop unroll(full)
      for (ushort col = 0; col < D_edge; col += 8) {
        float2 O = this->get(this->o(row, col));
        O *= l_inv;
        this->set(this->q(row, col), real2(O));
      }
    }
  }
  
  METAL_FUNC
  void store_o(threadgroup real* dst, bool maybe_skip_divide) {
    const bool skip_divide = maybe_skip_divide && !O_upcast;
    const ushort O_prep_offset = (skip_divide ? O_offset : Q_offset);
    
    auto _dst = dst + R_in_simd * D_edge + C_in_simd;
#pragma clang loop unroll(full)
    for (ushort row = 0; row < R_simd; row += 8) {
#pragma clang loop unroll(full)
      for (ushort col = 0; col < D_edge; col += 8) {
        auto matrix = _index(O_prep_offset, D_edge, row, col);
        vec_store2<
        real, threadgroup real*, threadgroup real2*, ushort
        >(*matrix, _dst, D_edge, ushort2(col, row));
      }
    }
  }
  
  // Stores data from the O matrix. Do not confuse the `index` argument with the
  // `offset` argument from other functions. Do not forget to append `sid.y *
  // C_simd` to the index before using this (also don't forget to append to the
  // threadgroup pointer when loading with C splits).
  METAL_FUNC
  void store_o_direct(uint index, device real* dst, bool maybe_skip_divide) {
    // This requires that R and D are multiples of 8. Otherwise, perform a two-
    // step store using threadgroup memory.
    if ((D % 8 != 0) || (R % 8 != 0)) {
      return;
    }
    const bool skip_divide = maybe_skip_divide && !O_upcast;
    const ushort R_edge = (R % R_group == 0) ? R_simd : (R % R_simd);
    const ushort O_prep_offset = (skip_divide ? O_offset : Q_offset);
    
    auto _dst = dst + index * D_edge;
#pragma clang loop unroll(full)
    for (ushort row = 0; row < R_edge; row += 8) {
#pragma clang loop unroll(full)
      for (ushort col = 0; col < D_edge; col += 8) {
        auto matrix = _index(O_prep_offset, D_edge, row, col);
        vec_store2<
        real, device real*, device real2*, uint
        >(*matrix, _dst, D_edge, ushort2(col, row));
      }
    }
    
    if ((R % R_group != 0) && (index + R_group < R)) {
#pragma clang loop unroll(full)
      for (ushort row = R_edge; row < R_simd; row += 8) {
#pragma clang loop unroll(full)
        for (ushort col = 0; col < D_edge; col += 8) {
          auto matrix = _index(O_prep_offset, D_edge, row, col);
          vec_store2<
          real, device real*, device real2*, uint
          >(*matrix, _dst, D_edge, ushort2(col, row));
        }
      }
    }
  }
};

// Perhaps try not immediately waiting on the SIMD future in low-occupancy
// situations. You would allocate twice as much threadgroup SRAM so V loads
// concurrently to K multiplication and vice versa. This might speed up GEMM as
// well (time division).
template <typename real>
struct threadgroup_allocation {
  threadgroup real* sram;
  
  METAL_FUNC
  threadgroup_allocation(threadgroup real* sram) {
    this->sram = sram;
  }
  
  // If this is the first block, we need to zero-initialize the padding on the
  // D dimension. It might be faster to write directly to threadgroup memory
  // from each thread, but we'll defer that optimization to another time.
  METAL_FUNC
  void load(uint index, device real* src, bool is_q, bool is_v)
  {
    ushort columns = (is_q ? D_edge : D);
    ushort rows;
    if (is_q) {
      if (R % R_group == 0) {
        rows = R_group;
      } else {
        rows = min(uint(R_group), R - index);
      }
    } else {
      if (C % C_group == 0) {
        rows = C_group;
      } else {
        rows = min(uint(C_group), C - index);
      }
    }
    
    ushort padded_rows = rows;
    if (is_v && (C % C_group != 0)) {
      // TODO: After debugging, elide the zero-padding when C > C_group. The
      // previous iteration filled the entire threadgroup memory with real
      // numbers. Although the data is garbage, it will multiply with 0 without
      // producing NAN. This optimization might not be applicable with C splits,
      // as the partial sums written to threadgroup memory can contain undefined
      // edge data.
      padded_rows = ~7 & (rows + 7);
    }
    
    __metal_simdgroup_event_t events[2];
    events[0] = __metal_simdgroup_async_copy_2d
    (
     sizeof(real), alignof(real),
     (threadgroup void*)sram, columns, 1, ulong2(columns, padded_rows),
     (device void*)(src + index * D), D, 1, ulong2(D, rows), long2(0, 0), 0);
    
    if (is_q && (C_group > R_group) && (D != D_edge)) {
      ulong2 padding_tile(D_edge - D, C_group - rows);
      events[1] = __metal_simdgroup_async_copy_2d
      (
       sizeof(real), alignof(real),
       (threadgroup void*)(sram + D), columns, 1, padding_tile,
       (device void*)src, D, 1, ulong2(0, 0), long2(0, 0), 0);
      __metal_wait_simdgroup_events(2, events);
    } else {
      __metal_wait_simdgroup_events(1, events);
    }
  }
  
  METAL_FUNC
  void store(uint index, device real* dst) {
    ushort rows;
    if (R % R_group == 0) {
      rows = R_group;
    } else {
      rows = min(uint(R_group), R - index);
    }
    
    auto _dst = dst + index * D;
    auto event = __metal_simdgroup_async_copy_2d
    (
     sizeof(real), alignof(real),
     (device void*)_dst, D, 1, ulong2(D, rows),
     (threadgroup void*)sram, D_edge, 1, ulong2(D, rows), long2(0, 0), 0);
    
    // We don't need to wait on the very last event.
    // WARNING: This approach may not work if you want to traverse multiple rows
    // with the same threadgroup.
//    __metal_wait_simdgroup_events(1, &event);
  }
};

// FlashAttention Metal kernel using threadgroup memory.
#if defined(__HAVE_SIMDGROUP_FUTURE__) || defined(SIMD_FUTURES_HIGHLIGHTING)
template <
typename real,
ushort matrices // number of SIMD matrices in the `thread` SRAM allocation
>
void _flash_attention
(
 device void *Q [[buffer(0)]],
 device void *K [[buffer(1)]],
 device void *V [[buffer(2)]],
 device real *O [[buffer(3)]],
 
 // This may have to change to `threadgroup void*` with mixed precision.
 threadgroup real *threadgroup_block [[threadgroup(0)]],
 
 // Y dimension ignored for now. It may be used for dispatching two threadgroups
 // per row in a variant of `flash_attention_triangular`. Triangular
 // FlashAttention will not have a batched version for the foreseeable future,
 // because of a lack of need and ambiguity about argument strides.
 //
 // The code must also be heavily modified for triangular FA; we can't hard-code
 // a value for the sequence length. This may warrant a new Metal shader file
 // with careful thought on how to de-duplicate code. If we find an optimal set
 // of hyperparameters (e.g. R splits, block size) from dense attention, some
 // constants could be hard-coded.
 uint2 gid [[threadgroup_position_in_grid]],
 ushort sidx [[simdgroup_index_in_threadgroup]],
 ushort lane_id [[thread_index_in_simdgroup]])
{
  // R_group must not exceed C_group.
  if (R_group > C_group) {
    return;
  }
  
  // All precisions must currently be the same.
  if ((Q_precision != K_precision) ||
      (Q_precision != V_precision) ||
      (Q_precision != O_precision)) {
    return;
  }
  
  // After debugging, try optimizations that fuse the generation of indices into
  // threadgroup memory.
  ushort quad_id = lane_id / 4;
  ushort R_in_simd = get_simdgroup_matrix_m(lane_id, quad_id);
  ushort C_in_simd = get_simdgroup_matrix_n(lane_id, quad_id);
  thread_allocation<real, matrices> thread_sram(R_in_simd, C_in_simd);
  
  ushort2 sid(sidx % C_splits, sidx / C_splits);
  ushort R_of_simd = sid.y * R_simd;
  ushort C_of_simd = sid.x * C_simd;
  threadgroup_allocation<real> threadgroup_sram(threadgroup_block);
  
  // Cache the Q block.
  {
    uint i = get_i(gid);
    if (sidx == 0) {
      threadgroup_sram.load(i, (device real*)Q, true, false);
    }
  }
  
  // Initialize the accumulator while waiting on the future's barrier.
  // TODO: Defer the preceding SIMD event wait in low-occupancy situations.
#pragma clang loop unroll(full)
  for (ushort r = 0; r < R_simd; r += 8) {
#pragma clang loop unroll(full)
    for (ushort d = 0; d < D_edge; d += 8) {
      *thread_sram.o(r, d) = simdgroup_float8x8(0);
    }
    *thread_sram.l(r) = 0;
    *thread_sram.m(r) = -INFINITY;
  }
  
  // Load the Q block.
  threadgroup_barrier(mem_flags::mem_threadgroup);
  auto Q_src = threadgroup_block + R_of_simd * D_edge;
  thread_sram.load(Q_offset, Q_src, ushort2(D_edge, R_simd));
  
  for (uint j = 0; j < C; j += C_group) {
    // Cache the K block.
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sidx == 0) {
      threadgroup_sram.load(j, (device real*)K, false, false);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Exit early from the last loop iteration.
    int C_trail_simd = int(C - j) - C_of_simd;
    if ((C % C_group != 0) && (C_trail_simd < 0)) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      threadgroup_barrier(mem_flags::mem_threadgroup);
      break;
    }
    
    // Multiply Q by K^T.
    auto _K_cache = threadgroup_block + C_of_simd * D_edge;
#pragma clang loop unroll(full)
    for (ushort d = 0; d < D_edge; d += 8) {
      // Load the K^T block.
      auto K_cache = _K_cache + d * C_group;
      thread_sram.load_transpose(KV_offset, K_cache, ushort2(8, C_simd));
      
#pragma clang loop unroll(full)
      for (ushort r = 0; r < R_simd; r += 8) {
        auto Q = thread_sram.q(r, d);
#pragma clang loop unroll(full)
        for (ushort c = 0; c < C_simd; c += 8) {
          auto K_T = thread_sram.k_transpose(d, c);
          auto S = thread_sram.s(r, c);
          
          if (d == 0) {
            simdgroup_multiply(*S, *Q, *K_T);
          } else {
            simdgroup_multiply_accumulate(*S, *Q, *K_T, *S);
          }
        }
      }
    }
    
    // Use `C_simd` instead of `C_group` for the check.
    const ushort C_edge = (C % C_simd == 0) ? C_simd : (C % C_simd);
    const ushort C_reduce_end = C_edge / 8 * 8;
    const ushort C_macc_end = (C_edge + 7) / 8 * 8;
    short C_trail_thread = short(C_edge) - C_of_simd - C_in_simd;
    
    // Perform the softmax.
    //
    // Fuse the multiplication `correction_new * P` with the generation of `l`.
    // We are upcasting to `float` and the downcasting would already incur an
    // extra instruction. It incurs that instruction because it's performed
    // after, not before, the sum. Therefore, we can fuse the correction factor
    // with the downcasting op for free. This also decreases numerical error and
    // allows the O_{corrected} + P * V multiplication to occur exclusively
    // through the SIMD MATMUL FMADD instruction.
    //
    // After explaining that optimization, we arrive at the next optimization.
    // Perform exp(S - m) using the new value for `m`. There's no need to change
    // it back into what would have been `exp(S - m_{ij})`. Instead, elide the
    // multiplication `exp(m_{ij} - m_i^{new}) * l_{ij}`. This has the downside
    // that we can't fuse the downcasting to FP16 with anything. It does reduce
    // the total number of computations and therefore the numerical error.
    //
    // Finally, we avoid the repetitive scaling by `l^{-1}` and restoring by `l`
    // the next iteration. Just store O as FP32 inside the registers. This
    // removes a potential need to scale P_{ij}V_{j} by `l^{-1}`. Register
    // pressure does not become a bottleneck (the Apple GPU has too much
    // registers and too little threadgroup memory). The only downside is,
    // initializing the accumulator takes twice as long.
#pragma clang loop unroll(full)
    for (ushort r = 0; r < R_simd; r += 8) {
      // m_{ij} = rowmax(S_{ij})
      real _m;
      thread_sram.reduce_max(r, { 0, C_reduce_end }, &_m);
      if (C % C_simd != 0) {
        if (C_trail_simd >= int(C_simd)) {
          thread_sram.reduce_max(r, { C_reduce_end, C_simd }, &_m);
        } else if (C_trail_thread > 0) {
          // Since fast math produces correct results with non-infinite numbers,
          // we use -MAXHALF or -MAXFLOAT.
          real mask = -numeric_limits<real>::max();
          auto S = thread_sram.get(thread_sram.s(r, C_reduce_end));
          thread_sram.apply_mask(&S, mask, C_trail_thread);
          
          real partial = max(S[0], S[1]);
          if (C_reduce_end == 0) {
            _m = partial;
          } else {
            _m = max(_m, partial);
          }
        }
      }
      
      float m = float(_m);
      if (is_function_constant_defined(QK_T_alpha)) {
        m *= QK_T_alpha;
      }
      m = fast::max(m, quad_shuffle_xor(m, 1));
      m = fast::max(m, simd_shuffle_xor(m, 8));
      
      // m_i^{new} = max(m_i, m_{ij})
      float m_old = *thread_sram.m(r);
      float m_new = fast::max(m_old, m);
      float sub = M_LOG2E_F * m_new;
      float correction = quick_exp(m_old, M_LOG2E_F, sub);
      *thread_sram.m(r) = m_new;
      
      // If all rows spanned by this SIMD don't have anything new, we can
      // elide the updating of the O matrix. There's only 8-way divergence, so
      // there's a good chance it will provide a measurable speedup.
      if (m_old != m_new) {
        // exp(m_i - m_i^{new}) * O_i
#pragma clang loop unroll(full)
        for (ushort d = 0; d < D_edge; d += 8) {
          auto O = thread_sram.get(thread_sram.o(r, d));
          O *= correction;
          thread_sram.set(thread_sram.o(r, d), O);
        }
      }
      
      // P_{ij} = exp(S_{ij} - m_{ij}) (pointwise)
      // l = rowsum(P)
      //
      // The first pass applies the elementwise operator, overshooting C_edge.
      // The second pass masks the edge values with 0.
      // The third pass gathers sum, overshooting C_edge.
      // These three passes are fused, so that intermediates remain in FP32.
      float l;
      thread_sram.reduce_sum(r, { 0, C_reduce_end }, &l, sub);
      if (C % C_simd != 0) {
        if (C_trail_simd >= int(C_simd)) {
          thread_sram.reduce_sum(r, { C_reduce_end, C_simd }, &l, sub);
        } else if (C_trail_thread > 0) {
          auto S = thread_sram.get(thread_sram.s(r, C_reduce_end));
          float2 P = {
            quick_exp(S[0], S_mult, sub),
            quick_exp(S[1], S_mult, sub)
          };
          
          // Before summation, we quickly zero out the edge of P. This fuses the
          // zero-masking for softmax + zero-padding the K dimension for GEMM.
          thread_sram.apply_mask(&P, 0, C_trail_thread);
          thread_sram.set(thread_sram.s(r, C_reduce_end), vec<real, 2>(P));
          
          float partial = P[0] + P[1];
          if (C_reduce_end == 0) {
            l = partial;
          } else {
            l += partial;
          }
        }
      }
      l += quad_shuffle_xor(m, 1);
      l += simd_shuffle_xor(m, 8);
      
      // l_i^{new} = exp(m_i - m_i^{new}) * l_i + l_{ij}
      float l_old = *thread_sram.l(r);
      *thread_sram.l(r) = fast::fma(l_old, correction, l);
    }
    
    // Cache the V block.
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sidx == 0) {
      threadgroup_sram.load(j, (device real*)V, false, false);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Multiply P by V.
    auto _V_cache = threadgroup_block + C_of_simd * D_edge;
#pragma clang loop unroll(full)
    for (ushort c = 0; c < C_macc_end; c += 8) {
      // Load the V block.
      auto V_cache = _V_cache + c * D_edge;
      thread_sram.load(KV_offset, V_cache, ushort2(D_edge, 8));
      
#pragma clang loop unroll(full)
      for (ushort r = 0; r < R_simd; r += 8) {
        auto P = thread_sram.p(r, c);
#pragma clang loop unroll(full)
        for (ushort d = 0; d < D_edge; d += 8) {
          auto V = thread_sram.v(c, d);
          auto O = thread_sram.o(r, d);
          simdgroup_multiply_accumulate(*O, *P, *V, *O);
        }
      }
    }
    if ((C % C_simd != 0) && (j + C_macc_end < C)) {
#pragma clang loop unroll(full)
      for (ushort c = C_macc_end; c < C_simd; c += 8) {
        // Load the V block.
        auto V_cache = _V_cache + c * D_edge;
        thread_sram.load(KV_offset, V_cache, ushort2(D_edge, 8));
        
#pragma clang loop unroll(full)
        for (ushort r = 0; r < R_simd; r += 8) {
          auto P = thread_sram.p(r, c);
#pragma clang loop unroll(full)
          for (ushort d = 0; d < D_edge; d += 8) {
            auto V = thread_sram.v(c, d);
            auto O = thread_sram.o(r, d);
            simdgroup_multiply_accumulate(*O, *P, *V, *O);
          }
        }
      }
    }
  }
  
  // Consolidate partial sums across C splits. Before sending O to another simd,
  // compress it into FP16 to reduce threadgroup bandwidth. On the other side,
  // it will expand back into FP32. This can be done with an FFMA: scale by the
  // other simd's `l_i` while adding to the current accumulator.
  //
  // This is a log(n) complexity parallel reduction. If C_splits > 2, the
  // threadgroup SRAM be at least as large as (C_splits / 2) * sizeof(Q_block).
  if (C_splits > 1) {
    // `C_reach` is how far to reach when asking for a sender simd. This is
    // analogous to the second argument of `quad/simd_shuffle_xor`.
    for (ushort C_reach = C_splits / 2; C_reach > 0; C_reach /= 2) {
      // Locate the threadgroup memory to write to.
      bool return_early;
      threadgroup real *message;
      if (C_splits == 2) {
        return_early = (sid.x != 0);
        message = threadgroup_block;
      } else {
        ushort receiver = ~(C_reach - 1) & sid.x;
        return_early = (sid.x != receiver);
        message = threadgroup_block + receiver * R_simd * D_edge;
      }
      
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (return_early) {
        // Sending simds: write `m` and `l` to threadgroup memory.
        auto threadgroup_lm = (threadgroup float2*)message;
#pragma clang loop unroll(full)
        for (ushort r = 0; r < R_simd; r += 8) {
          auto l = *thread_sram.l(r);
          auto m = *thread_sram.m(r);
          threadgroup_lm[r + R_in_simd] = float2(l, m);
        }
        
        // Sending simds: prepare half of O.
        thread_sram.prepare_o(true, true);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      
      if (return_early) {
        // Sending simds: prepare half of O.
        thread_sram.prepare_o(false, true);
      } else {
        // Receiving simds: read `m` and `l`.
        auto threadgroup_lm = (threadgroup float2*)message;
        ushort lane_modulo = lane_id % 2;
#pragma clang loop unroll(full)
        for (ushort r = 0; r < R_simd; r += 8) {
          float2 sender_lm = threadgroup_lm[r + R_in_simd];
          *thread_sram.l_other(r) = sender_lm[0];
          *thread_sram.m_other(r) = sender_lm[1];
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      
      if (return_early) {
        // Sending simds: write O.
        thread_sram.store_o(message, true);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      
      if (return_early) {
        // Sending simds: return from the function. Note that this approach
        // won't work if you want to reuse the simds for more computation later.
        return;
      } else {
        // We could optimize this a little by incrementing the O base address a
        // constant amount every iteration, instead of loading a new
        // `r * D_edge` value from the instruction cache. This optimization is
        // not a priority right now.
        threadgroup real *O_base_addr = message;
        O_base_addr += (R_of_simd + R_in_simd) * D_edge;
        O_base_addr += C_in_simd;
        
        // Receiving simds: update O.
#pragma clang loop unroll(full)
        for (ushort r = 0; r < R_simd; r += 8) {
          float m = *thread_sram.m_other(r);
          float m_old = *thread_sram.m(r);
          float m_new = fast::max(m_old, m);
          float sub = M_LOG2E_F * m_new;
          *thread_sram.m(r) = m_new;
          
          float2 corrections = {
            quick_exp(m_old, M_LOG2E_F, sub),
            quick_exp(m, M_LOG2E_F, sub)
          };
          float l = *thread_sram.l_other(r);
          float l_old = *thread_sram.l(r);
          float l_new = fast::fma(corrections[0], l_old, corrections[1] * l);
          *thread_sram.l(r) = l_new;
          
          // Undo the range reduction, upcasting the sender's data to FP32.
          if (O_upcast) {
            corrections[1] *= l;
          }
          
          // Hold the O data live inside the register cache. Although this
          // approach can't elide the multiplication by a correction factor for
          // either operand, it might be free in register bandwidth-bound FP32.
          threadgroup real *O_addr = O_base_addr + r * D_edge;
#pragma clang loop unroll(full)
          for (ushort d = 0; d < D_edge; d += 8) {
            // Load O_sender before updating 'O', to hopefully hint that the
            // compiler should hide this latency.
            typedef vec<real, 2> real2;
            real2 O_sender = *(threadgroup real2*)(O_addr + d);
            float2 O = thread_sram.get(thread_sram.o(r, d));
            O *= corrections[0];
            
            O[0] = fast::fma(float(O[0]), corrections[1], real(O_sender[0]));
            O[1] = fast::fma(float(O[1]), corrections[1], real(O_sender[1]));
            thread_sram.set(thread_sram.o(r, d), O);
          }
        }
      }
    }
  }
  
  // Write the accumulator back to memory.
  thread_sram.prepare_o(true, false);
  thread_sram.prepare_o(false, false);
  {
    uint i = get_i(gid);
    if ((D % 8 == 0) && (R % 8 == 0)) {
      thread_sram.store_o_direct(i, O, false);
    } else {
      thread_sram.store_o(threadgroup_block, false);
      threadgroup_sram.store(i, O);
    }
  }
}

kernel void flash_attention
(
 device void *Q [[buffer(0)]],
 device void *K [[buffer(1)]],
 device void *V [[buffer(2)]],
 device void *O [[buffer(3)]],
 threadgroup void *threadgroup_block [[threadgroup(0)]],
 
 uint2 gid [[threadgroup_position_in_grid]],
 ushort sidx [[simdgroup_index_in_threadgroup]],
 ushort lane_id [[thread_index_in_simdgroup]])
{
  // Only check Q precision here; the function body ensures K, V, and O match.
  if (Q_precision == metal_flash_attention::HALF) {
    
  } else if (Q_precision == metal_flash_attention::FLOAT) {
    
  }
}

kernel void flash_attention_batched
(
 device void *Q [[buffer(0)]],
 device void *K [[buffer(1)]],
 device void *V [[buffer(2)]],
 device void *O [[buffer(3)]],
 threadgroup void *threadgroup_block [[threadgroup(0)]],
 
 uint3 gid [[threadgroup_position_in_grid]],
 ushort sidx [[simdgroup_index_in_threadgroup]],
 ushort lane_id [[thread_index_in_simdgroup]])
{
  // Only check Q precision here; the function body ensures K, V, and O match.
  if (Q_precision == metal_flash_attention::HALF) {
    auto _Q = (device half*)Q + gid.z * Q_stride;
    auto _K = (device half*)K + gid.z * K_stride;
    auto _V = (device half*)V + gid.z * V_stride;
    auto _O = (device half*)O + gid.z * O_stride;
    auto _block = (threadgroup half*)threadgroup_block;
    _flash_attention<
    half, 1024
    >(_Q, _K, _V, _O, _block, gid.xy, sidx, lane_id);
  } else if (Q_precision == metal_flash_attention::FLOAT) {
    auto _Q = (device float*)Q + gid.z * Q_stride;
    auto _K = (device float*)K + gid.z * K_stride;
    auto _V = (device float*)V + gid.z * V_stride;
    auto _O = (device float*)O + gid.z * O_stride;
    auto _block = (threadgroup float*)threadgroup_block;
    _flash_attention<
    float, 1024
    >(_Q, _K, _V, _O, _block, gid.xy, sidx, lane_id);
  }
}
#endif

#endif
