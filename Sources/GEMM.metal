//
//  GEMM.metal
//  MetalFlashAttention
//
//  Created by Philip Turner on 6/23/23.
//

#include <metal_stdlib>
#include "metal_simdgroup_event"
#include "metal_simdgroup_matrix_storage"
using namespace metal;

// template <typename T> void gemm(uint2 threadgroup_position_in_grid)

// kernel void hgemm(uint3 threadgroup_position_in_grid)

// kernel void sgemm(uint3 threadgroup_position_in_grid)
