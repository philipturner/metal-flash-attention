//
//  Attention.metal
//  MetalFlashAttention
//
//  Created by Philip Turner on 6/26/23.
//

#include <metal_stdlib>
#include "metal_data_type"
#include "metal_simdgroup_event"
#include "metal_simdgroup_matrix_storage"
using namespace metal;

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused"

// Dimensions of each matrix.
constant uint R [[function_constant(0)]];
constant uint C [[function_constant(1)]];
constant uint H [[function_constant(2)]];
constant uint D [[function_constant(3)]];

// Whether each matrix is transposed.
constant bool Q_trans [[function_constant(10)]];
constant bool K_trans [[function_constant(11)]];
constant bool V_trans [[function_constant(12)]];
constant bool O_trans [[function_constant(13)]];
constant bool mask_trans = false;
constant bool lm_trans = true;

// Leading dimension for the operand and its gradient.
constant uint Q_leading_dim = Q_trans ? R : H * D;
constant uint K_leading_dim = K_trans ? H * D : C;
constant uint V_leading_dim = V_trans ? C : H * D;
constant uint O_leading_dim = O_trans ? R : H * D;
constant uint mask_leading_dim = mask_trans ? R : C;
constant uint lm_leading_dim = lm_trans ? R : H * D;

// Data type of each matrix.
constant uint Q_data_type [[function_constant(20)]];
constant uint K_data_type = Q_data_type;
constant uint V_data_type = Q_data_type;
constant uint O_data_type = Q_data_type;
constant uint mask_data_type [[function_constant(24)]];

constant uint dQ_data_type = (Q_data_type == MTLDataTypeFloat) ? MTLDataTypeFloat : MTLDataTypeBFloat;
constant uint dK_data_type = dQ_data_type;
constant uint dV_data_type = dQ_data_type;
constant uint dO_data_type = dQ_data_type;
constant uint lm_data_type = MTLDataTypeFloat;

constant bool batched [[function_constant(100)]];
constant bool forward [[function_constant(101)]];
constant bool backward [[function_constant(102)]];
constant bool gradient [[function_constant(103)]];

constant bool masked [[function_constant(110)]];
constant bool block_sparse [[function_constant(111)]];
constant bool triangular [[function_constant(112)]];
constant bool store_lm = (gradient && forward) || triangular;
constant bool generate_block_mask = block_sparse && !forward && !backward;
// TODO: Change l -> inv(l) after combining partial sums for triangular.

constant ushort R_simd [[function_constant(200)]];
constant ushort C_simd [[function_constant(201)]];

constant ushort R_splits [[function_constant(210)]];
constant ushort C_splits = 1;

// TODO: To avoid zero-padding for D, create a single SIMD matrix in registers,
// which masks the tail elements.
constant ushort R_group = R_simd * R_splits;
constant ushort C_group = C_simd * C_splits;
constant ushort D_padded = (D + 7) / 8 * 8;

constant ushort Q_block_leading_dim = (Q_trans ? R_group : D);
constant ushort K_block_leading_dim = (K_trans ? D : C_group);
constant ushort V_block_leading_dim = (V_trans ? C_group : D);
constant ushort O_block_leading_dim = (O_trans ? R_group : D);
constant ushort mask_block_leading_dim = (mask_trans ? R_group : C_group);

#pragma clang diagnostic pop

// TODO:
// Grid X = C dimension
// Grid Y = R dimension
// Grid Z = batch dimension
// Threadgroup size = 32
template <typename T>
void _generate_block_mask_impl()
{
  
}

// TODO:
// Grid X = sequence dimension (R for forward, C for backward)
// Grid Y = heads dimension (usually 1-8)
// Grid Z = batch dimension
// Threadgroup size = 128
template <typename T, typename dT>
void _attention_impl()
{
  // For every element in threadgroup memory, there's 4-6 in registers.
  simdgroup_matrix_storage<T> Q_sram[1024];
  simdgroup_matrix_storage<float> O_sram[1024];
  simdgroup_matrix_storage<float> dK_sram[1024];
  simdgroup_matrix_storage<float> dV_sram[1024];
}

kernel void attention()
{
  if (Q_data_type == MTLDataTypeFloat) {
    _attention_impl<float, float>();
  } else {
    _attention_impl<half, ushort>();
  }
}
