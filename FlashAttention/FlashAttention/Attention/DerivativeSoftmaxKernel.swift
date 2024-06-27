//
//  DerivativeSoftmaxKernel.swift
//  FlashAttention
//
//  Created by Philip Turner on 6/26/24.
//

import Metal

// Fuses the generation of dP (GEMM) with the generation of dS (elementwise).
struct DerivativeSoftmaxKernel {
  // The source code to compile.
  var source: String = ""
  
  // A copy of the block dimensions from the descriptor.
  var blockDimensions: (M: UInt16, N: UInt16, K: UInt16)
  
  // If you allocate threadgroup memory after compiling the kernel, the code
  // has higher performance.
  var threadgroupMemoryAllocation: UInt16
  
  // The number of threads per group.
  var threadgroupSize: UInt16
  
  init(descriptor: GEMMKernelDescriptor) {
    guard let blockDimensions = descriptor.blockDimensions,
          let memoryPrecisions = descriptor.memoryPrecisions,
          let registerPrecisions = descriptor.registerPrecisions,
          let splits = descriptor.splits,
          let transposeState = descriptor.transposeState else {
      fatalError("Descriptor was incomplete: \(descriptor)")
    }
    self.blockDimensions = blockDimensions
    self.threadgroupSize = 32 * splits.M * splits.N
    
    // Validate the correctness of the dO precision.
    switch (memoryPrecisions.A, registerPrecisions.A) {
    case (.FP32, .FP32):
      // 32-bit training.
      break
    case (.BF16, .FP32):
      // Mixed precision training (M1).
      break
    case (.BF16, .BF16):
      // Mixed precision training (M3).
      break
    default:
      fatalError("Operand A had an invalid precision.")
    }
    
    // Validate the correctness of the V precision.
    switch (memoryPrecisions.B, registerPrecisions.B) {
    case (.FP32, .FP32):
      // 32-bit training.
      break
    case (.FP16, .FP16):
      // Mixed precision training.
      break
    default:
      fatalError("Operand B had an invalid precision.")
    }
    
    // Validate the correctness of the dS precision.
    switch (memoryPrecisions.C, registerPrecisions.C) {
    case (.FP32, .FP32):
      // 32-bit training.
      break
    case (.BF16, .FP32):
      // Mixed precision training.
      break
    default:
      fatalError("Operand C had an invalid precision.")
    }
    
    // Inject the contents of the headers.
    source += """
\(createMetalSimdgroupEvent())
\(createMetalSimdgroupMatrixStorage())
using namespace metal;

"""
    
    // Declare the size of M and N within a register allocation.
    let registerM: UInt16 = blockDimensions.M / splits.M
    let registerN: UInt16 = blockDimensions.N / splits.N
    
    // Retrieve the "padded" block dimensions, otherwise compute analytically
    // from the true block dimensions.
    var paddedBlockDimensionsA: (M: UInt16, K: UInt16)
    var paddedBlockDimensionsB: (K: UInt16, N: UInt16)
    var paddedBlockDimensionsC: (M: UInt16, N: UInt16)
    if let paddedBlockDimensions = descriptor.paddedBlockDimensions {
      paddedBlockDimensionsA = paddedBlockDimensions.A
      paddedBlockDimensionsB = paddedBlockDimensions.B
      paddedBlockDimensionsC = paddedBlockDimensions.C
    } else {
      paddedBlockDimensionsA = (blockDimensions.M, blockDimensions.K)
      paddedBlockDimensionsB = (blockDimensions.K, blockDimensions.N)
      paddedBlockDimensionsC = (blockDimensions.M, blockDimensions.N)
    }
    
    // Determine the block dimensions from the transpose state.
    var leadingDimensionA: String
    var leadingDimensionB: String
    var leadingBlockDimensionA: UInt16
    var leadingBlockDimensionB: UInt16
    if transposeState.A {
      leadingDimensionA = "M"
      leadingBlockDimensionA = paddedBlockDimensionsA.M
    } else {
      leadingDimensionA = "K"
      leadingBlockDimensionA = paddedBlockDimensionsA.K
    }
    if transposeState.B {
      leadingDimensionB = "K"
      leadingBlockDimensionB = paddedBlockDimensionsB.K
    } else {
      leadingDimensionB = "N"
      leadingBlockDimensionB = paddedBlockDimensionsB.N
    }
    
    // Add the function constants.
    do {
      source += """

// Dimensions of each matrix.
// - Limitations to matrix size:
//   - 2^32 in each dimension (M/N/K).
//   - Extending to 2^64 may require changing 'uint' to 'ulong'. There is a
//     good chance this will significantly degrade performance, and require
//     changing the data type of several variables that process addresses. The
//     client is responsible for ensuring correctness and performance with
//     matrices spanning several billion elements in one direction.
//   - The matrix dimensions must be known at compile time, via function
//     constants. Dynamic matrix shapes are beyond the scope of this reference
//     implementation. Dynamic shapes cause a non-negligible regression to
//     shader execution speed. However, they could minimize a compilation
//     latency bottleneck in some use cases.
// - Limitations to batch size:
//   - Dictated by how the client modifies the code to implement batching.
//   - Dynamic batch shapes would likely not harm performance much. For example,
//     someone could enter an array of pointers/memory offsets to different
//     matrices in the batch. Each slice of a 3D thread grid could read a
//     different pointer from memory, and use that pointer as the A/B/C matrix.
//     Another approach is to restrict the input format, so all matrices are
//     stored contiguously in memory. Then, the memory offset could be computed
//     analytically from matrix size and the Z dimension in a 3D thread grid.
//
// Another note:
// - The rows of the matrix must be contiguous in memory. Supporting strides
//   that differ from the actual matrix dimensions should not be difficult, but
//   it is out of scope for this reference kernel.
constant uint M [[function_constant(0)]];
constant uint N [[function_constant(1)]];
constant uint K [[function_constant(2)]];

// Whether each matrix is transposed.
constant bool A_trans = \(transposeState.A);
constant bool B_trans = \(transposeState.B);

// Define the memory layout of the matrix block.
constant ushort M_group = \(blockDimensions.M);
constant ushort N_group = \(blockDimensions.N);
constant ushort K_group = \(blockDimensions.K);

// Thresholds that mark the matrix edge.
constant uint M_edge = M - (M % M_group);
constant uint N_edge = N - (N % N_group);

// Find the number of elements in the final block. If the matrix
// dimensions are perfectly divisibly by block dimensions, we don't want
// this value to be zero. The final block is a full block.
constant ushort M_remainder = (M % \(registerM) == 0)
  ? \(registerM) : M % \(registerM);
constant ushort N_remainder = (N % \(registerN) == 0)
  ? \(registerN) : N % \(registerN);
constant ushort K_remainder = (K % K_group == 0)
  ? K_group : K % K_group;
constant ushort K_remainder_padded = (K_remainder + 7) / 8 * 8;

// Shift the final block, so it doesn't access out-of-bounds memory.
constant ushort M_shift = (M < M_group) ? 0 : \(registerM) - M_remainder;
constant ushort N_shift = (N < N_group) ? 0 : \(registerN) - N_remainder;
"""
    }
    
    // Allocate threadgroup memory, using the 'memory precision'. This memory
    // is allocated at runtime, either by the user (explicit API call) or by
    // the driver (behind the scenes).
    let memoryNameA = memoryPrecisions.A.name
    let memoryNameB = memoryPrecisions.B.name
    let memoryNameC = memoryPrecisions.C.name
    
    // Allocate thread memory, using the 'register precision'. This memory
    // is allocated by embedding the precision into the assembly code.
    let registerNameA = registerPrecisions.A.name
    let registerNameB = registerPrecisions.B.name
    
    // Add the utility functions.
    source += """

// The layout of threads within a SIMD matrix.
//
//  0  0  1  1  8  8  9  9
//  2  2  3  3 10 10 11 11
//  4  4  5  5 12 12 13 13
//  6  6  7  7 14 14 15 15
// 16 16 17 17 24 24 25 25
// 18 18 19 19 26 26 27 27
// 20 20 21 21 28 28 29 29
// 22 22 23 23 30 30 31 31
//
// This is Morton order, a method for coalescing data accesses. It is used
// in a variety of contexts, from ray tracing acceleration structures, to
// nodal-point Laplacians, to sorting large lattices of atoms.
//
// Source: https://patents.google.com/patent/US11256518B2
METAL_FUNC ushort2 morton_order(ushort thread_index_in_simdgroup) {
  ushort lane_id = thread_index_in_simdgroup;
  ushort quad_id = lane_id / 4;
  
  constexpr ushort QUADRANT_SPAN_M = 4;
  constexpr ushort THREADS_PER_QUADRANT = 8;
  ushort M_floor_of_quadrant = (quad_id / 4) * QUADRANT_SPAN_M;
  ushort M_in_quadrant = (lane_id / 2) % (THREADS_PER_QUADRANT / 2);
  ushort M_in_simd = M_floor_of_quadrant + M_in_quadrant;
  
  ushort N_floor_of_quadrant = (quad_id & 2) * 2; // 0 or 4
  ushort N_in_quadrant = (lane_id % 2) * 2; // 0 or 2
  ushort N_in_simd = N_floor_of_quadrant + N_in_quadrant;
  
  return ushort2(N_in_simd, M_in_simd);
}

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
  const \(addressSpace) \(memoryNameA) *A_src,
  const \(addressSpace) \(memoryNameB) *B_src,
  thread simdgroup_matrix_storage<\(registerNameA)> *A_sram,
  thread simdgroup_matrix_storage<\(registerNameB)> *B_sram,
  thread simdgroup_matrix_storage<float> *C_sram,
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
      multiplyDesc.leadingDimensionA = leadingDimensionA
      multiplyDesc.leadingDimensionB = leadingDimensionB
      source += createMultiply(descriptor: multiplyDesc)
      
      multiplyDesc.addressSpace = "threadgroup"
      multiplyDesc.leadingDimensionA = "\(leadingBlockDimensionA)"
      multiplyDesc.leadingDimensionB = "\(leadingBlockDimensionB)"
      source += createMultiply(descriptor: multiplyDesc)
    }
    
    // Add the setup portion where the addresses are prepared.
    do {
      var blockBytesA = paddedBlockDimensionsA.M * paddedBlockDimensionsA.K
      var blockBytesB = paddedBlockDimensionsB.K * paddedBlockDimensionsB.N
      var blockBytesC = paddedBlockDimensionsC.M * paddedBlockDimensionsC.N
      blockBytesA *= UInt16(memoryPrecisions.A.size)
      blockBytesB *= UInt16(memoryPrecisions.B.size)
      blockBytesC *= UInt16(memoryPrecisions.C.size)
      threadgroupMemoryAllocation = max(blockBytesA + blockBytesB, blockBytesC)
      
      source += """

// Metal function arguments.
//
// A: the left-hand side matrix
// - dimensions: M x K
//               K x M (transposed)
// - memory precision: memA
// - register precision: regA
//
// B: the right-hand side matrix
// - dimensions: K x N
//               N x K (transposed)
// - memory precision: memB
// - register precision: regB
//
// C: the output matrix, alternatively the dot product accumulator
// - dimensions: M x N
// - memory precision: memC
// - register precision: regC
//
// threadgroup_block: the chunk of threadgroup memory allocated at runtime
// - ideally 10 KB or less
// - precision: void/8-bit integer to make the pointer arithmetic more legible
kernel void gemm(device \(memoryNameA) *A [[buffer(0)]],
                 device \(memoryNameB) *B [[buffer(1)]],
                 device \(memoryNameC) *C [[buffer(2)]],
                 
                 threadgroup uchar *threadgroup_block [[threadgroup(0)]],
                 
                 uint3 gid [[threadgroup_position_in_grid]],
                 ushort sidx [[simdgroup_index_in_threadgroup]],
                 ushort lane_id [[thread_index_in_simdgroup]])
{
  auto A_block = (threadgroup \(memoryNameA)*)(threadgroup_block);
  auto B_block = (threadgroup \(memoryNameB)*)(threadgroup_block + \(blockBytesA));
  ushort2 sid(sidx % \(splits.N), sidx / \(splits.N));
  ushort2 morton_offset = morton_order(lane_id);
  
  // Return early if the SIMD is out of bounds.
  //
  // There could be some threadgroups where the matrix edge cuts straight
  // through the middle of the block. SIMDs on the right or bottom of the
  // dividing line must be stopped from causing out-of-bounds accesses. This is
  // the reason for the early exit.
  uint M_offset = gid.y * M_group;
  uint N_offset = gid.x * N_group;
  {
    if (M_offset + sid.y * \(registerM) >= M ||
        N_offset + sid.x * \(registerN) >= N) {
      return;
    }
  }
  ushort2 offset_in_group(sid.x * \(registerN) + morton_offset.x,
                          sid.y * \(registerM) + morton_offset.y);
  
  // Shift the matrix block within bounds, if possible.
  if ((M_shift != 0) && (gid.y * M_group >= M_edge)) {
    M_offset -= M_shift;
  }
  if ((N_shift != 0) && (gid.x * N_group >= N_edge)) {
    N_offset -= N_shift;
  }

"""
    }
    
    // Add the setup of the accumulator.
    do {
      let arrayElementsC: UInt16 = (registerM / 8) * (registerN / 8)
      
      source += """

  simdgroup_matrix_storage<float> C_sram[\(arrayElementsC)];
  
  // Initialize the accumulator.
#pragma clang loop unroll(full)
  for (ushort m = 0; m < \(registerM); m += 8) {
#pragma clang loop unroll(full)
    for (ushort n = 0; n < \(registerN); n += 8) {
      ushort2 origin(n, m);
      auto C = get_sram(C_sram, \(registerN), origin);
      *C = simdgroup_matrix_storage<float>(0);
    }
  }

"""
    }
    
    // Add the matrix multiplication iterations.
    //
    // Async copies are required for correct behavior in edge cases. We attempt
    // to execute most iterations without async copy, and only the necessary
    // ones with async copy.
    do {
      var asyncIterationsStart: String
      if descriptor.preferAsyncLoad {
        asyncIterationsStart = "0"
      } else {
        asyncIterationsStart = "(K - (K % K_group))"
      }
      let paddedCeilingK = "(K + K_remainder_padded - K_remainder)"
      
      source += """

  // Perform the iterations where async copy is avoided.
  for (uint k = 0; k < \(asyncIterationsStart); k += 8) {
    uint2 A_offset(k, M_offset);
    uint2 B_offset(N_offset, k);
    A_offset += uint2(morton_offset.x, offset_in_group.y);
    B_offset += uint2(offset_in_group.x, morton_offset.y);
    
    auto A_src = simdgroup_matrix_storage<\(memoryNameA)>::apply_offset(
      A, \(leadingDimensionA), A_offset, A_trans);
    auto B_src = simdgroup_matrix_storage<\(memoryNameB)>::apply_offset(
      B, \(leadingDimensionB), B_offset, B_trans);

    simdgroup_matrix_storage<\(registerNameA)> A_sram[\(registerM / 8) * (8 / 8)];
    simdgroup_matrix_storage<\(registerNameB)> B_sram[(8 / 8) * \(registerN / 8)];
    multiply_accumulate(A_src, B_src,
                        A_sram, B_sram, C_sram, 0);
  }

  // Perform the iterations where async copy is used.
  for (uint k = \(asyncIterationsStart); k < K; k += K_group) {
    // Launch an async copy from device to threadgroup memory.
    if (sidx == 0) {
      uint2 A_offset(k, M_offset);
      uint2 B_offset(N_offset, k);
      auto A_src = simdgroup_matrix_storage<\(memoryNameA)>::apply_offset(
        A, \(leadingDimensionA), A_offset, A_trans);
      auto B_src = simdgroup_matrix_storage<\(memoryNameB)>::apply_offset(
        B, \(leadingDimensionB), B_offset, B_trans);

      ushort M_tile_dimension = min(uint(M_group), M - M_offset);
      ushort N_tile_dimension = min(uint(N_group), N - N_offset);
      ushort K_tile_dimension = min(uint(K_group), K - k);
      ushort K_tile_padded = min(uint(K_group), \(paddedCeilingK) - k);

      ushort2 A_tile_src(K_tile_dimension, M_tile_dimension);
      ushort2 B_tile_src(N_tile_dimension, K_tile_dimension);
      ushort2 A_tile_dst(K_tile_padded, M_tile_dimension);
      ushort2 B_tile_dst(N_tile_dimension, K_tile_padded);

      simdgroup_event events[2];
      events[0].async_copy(A_block, \(leadingBlockDimensionA), A_tile_dst,
                           A_src, \(leadingDimensionA), A_tile_src, A_trans);
      events[1].async_copy(B_block, \(leadingBlockDimensionB), B_tile_dst,
                           B_src, \(leadingDimensionB), B_tile_src, B_trans);
      simdgroup_event::wait(2, events);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    ushort2 A_block_offset(morton_offset.x, offset_in_group.y);
    ushort2 B_block_offset(offset_in_group.x, morton_offset.y);
    auto A_block_src = simdgroup_matrix_storage<\(memoryNameA)>::apply_offset(
      A_block, \(leadingBlockDimensionA), A_block_offset, A_trans);
    auto B_block_src = simdgroup_matrix_storage<\(memoryNameB)>::apply_offset(
      B_block, \(leadingBlockDimensionB), B_block_offset, B_trans);

    simdgroup_matrix_storage<\(registerNameA)> A_sram[\(registerM / 8) * (K_group / 8)];
    simdgroup_matrix_storage<\(registerNameB)> B_sram[(K_group / 8) * \(registerN / 8)];
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
    
    // Overwrite the accumulator with dS.
    do {
      var memoryNameP: String
      if memoryPrecisions.C == .BF16 {
        memoryNameP = "half"
      } else {
        memoryNameP = "float"
      }
      
      source += """

if ((M >= M_group) && (N >= N_group)) {
  uint2 C_offset(N_offset + offset_in_group.x,
                 M_offset + offset_in_group.y);
  auto C_src = simdgroup_matrix_storage<\(memoryNameP)>::apply_offset(
    (device \(memoryNameP)*)C, N, C_offset);
  
  // Write the accumulator to device memory.
#pragma clang loop unroll(full)
  for (ushort m = 0; m < \(registerM); m += 8) {
#pragma clang loop unroll(full)
    for (ushort n = 0; n < \(registerN); n += 8) {
      ushort2 origin(n, m);
      auto dP = get_sram(C_sram, \(registerN), origin);
      
      // dS = P * (dP - D);
      simdgroup_matrix_storage<\(memoryNameP)> P;
      P.load(C_src, N, origin);
      auto P_elements = *(P.thread_elements());
      *(dP->thread_elements()) *= float2(P_elements);
    }
  }
} else {
  // For simplicity, require that all matrices are large enough to employ the
  // shifting optimization. This means the sequence length must be >= 32.
  return;
}

"""
    }
    
    // Store the accumulator.
    do {
      var storeFunctionC: String
      if memoryPrecisions.C == .BF16 {
        storeFunctionC = "store_bfloat"
      } else {
        storeFunctionC = "store"
      }
      
      source += """

{
  // Always take the slow path. The shifting optimization will produce
  // incorrect results when two conflicting threads write the same value.
  auto C_block = (threadgroup \(memoryNameC)*)(threadgroup_block);
  auto C_block_dst = simdgroup_matrix_storage<\(memoryNameC)>::apply_offset(
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
    auto C_dst = simdgroup_matrix_storage<\(memoryNameC)>::apply_offset(
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
      C_block = simdgroup_matrix_storage<\(memoryNameC)>::apply_offset(
        C_block, N_group, C_block_shift);
    }
    
    simdgroup_event event;
    event.async_copy(C_dst, N, C_tile, C_block, N_group, C_tile);
  }
}
"""
    }
    
    // Add the final closing brace of the Metal function.
    source += "}" + "\n"
  }
  
  static func createPipeline(
    source: String,
    matrixDimensions: (M: UInt32, N: UInt32, K: UInt32)
  ) -> MTLComputePipelineState {
    let device = MTLContext.global.device
    let library = try! device.makeLibrary(source: source, options: nil)
    
    // Set the function constants.
    let constants = MTLFunctionConstantValues()
    var M = matrixDimensions.M
    var N = matrixDimensions.N
    var K = matrixDimensions.K
    constants.setConstantValue(&M, type: .uint, index: 0)
    constants.setConstantValue(&N, type: .uint, index: 1)
    constants.setConstantValue(&K, type: .uint, index: 2)
    
    let function = try! library.makeFunction(
      name: "gemm", constantValues: constants)
    let pipeline = try! device.makeComputePipelineState(function: function)
    return pipeline
  }
}
