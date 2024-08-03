//
//  GEMMKernel.swift
//  FlashAttention
//
//  Created by Philip Turner on 6/21/24.
//

import protocol Metal.MTLLibrary

struct GEMMKernel {
  // Compiled source code.
  var source: String = ""
  var library: MTLLibrary!
  
  // Address spaces and data types.
  var memoryPrecisions: (
    A: GEMMOperandPrecision, B: GEMMOperandPrecision, C: GEMMOperandPrecision)
  var preferAsyncLoad: Bool
  var preferAsyncStore: Bool
  var registerPrecisions: (
    A: GEMMOperandPrecision, B: GEMMOperandPrecision, C: GEMMOperandPrecision)
  
  // Block sizes.
  var blockBytes: (A: UInt16, B: UInt16, C: UInt16)
  var blockDimensions: (M: UInt16, N: UInt16, K: UInt16)
  var leadingDimensions: (A: String, B: String)
  var leadingBlockDimensions: (A: UInt16, B: UInt16)
  var registerM: UInt16
  var registerN: UInt16
  var threadgroupSize: UInt16
  
  init(descriptor: GEMMKernelDescriptor) {
    guard let blockDimensions = descriptor.blockDimensions,
          let device = descriptor.device,
          let memoryPrecisions = descriptor.memoryPrecisions,
          let preferAsyncStore = descriptor.preferAsyncStore,
          let registerPrecisions = descriptor.registerPrecisions,
          let splits = descriptor.splits,
          let transposeState = descriptor.transposeState else {
      fatalError("Descriptor was incomplete: \(descriptor)")
    }
    
    self.blockDimensions = blockDimensions
    self.memoryPrecisions = memoryPrecisions
    self.preferAsyncLoad = descriptor.preferAsyncLoad
    self.preferAsyncStore = preferAsyncStore
    self.registerPrecisions = registerPrecisions
    self.threadgroupSize = 32 * splits.M * splits.N
    
    // Validate the correctness of register precisions.
    func checkOperandPair(
      memory: GEMMOperandPrecision,
      register: GEMMOperandPrecision
    ) -> Bool {
      // Truth table:
      //
      // memory | register | valid |
      // ------ | -------- | ----- |
      // FP32   | FP32     | yes   |
      // FP32   | FP16     | no    |
      // FP32   | BF16     | no    |
      // FP16   | FP32     | yes   |
      // FP16   | FP16     | yes   |
      // FP16   | BF16     | no    |
      // BF16   | FP32     | yes   |
      // BF16   | FP16     | no    |
      // BF16   | BF16     | yes   |
      //
      // Optimized form of the logic:
      //
      // If the register precision matches the memory precision,
      //   return true
      // If the register precision equals FP32,
      //   return true
      // Otherwise,
      //   return false
      //
      // The logic statements will change if you introduce custom quantized
      // formats. The truth table will grow exponentially. You'll need to add
      // more restrictions on accepted pairs to overcome the combinatorial
      // explosion.
      if register == memory {
        return true
      } else if register == .FP32 {
        return true
      } else {
        return false
      }
    }
    
    guard checkOperandPair(
      memory: memoryPrecisions.A, register: registerPrecisions.A) else {
      fatalError("Operand A had an invalid register precision.")
    }
    guard checkOperandPair(
      memory: memoryPrecisions.B, register: registerPrecisions.B) else {
      fatalError("Operand B had an invalid register precision.")
    }
    guard checkOperandPair(
      memory: memoryPrecisions.C, register: registerPrecisions.C) else {
      fatalError("Operand C had an invalid register precision.")
    }
    if registerPrecisions.C == .BF16 {
      // BF16 has too few mantissa bits to be an accurate accumulator. In
      // addition, switching from FP32 accumulator to BF16 accumulator slows
      // down execution speed on both M1/M2 and M3+.
      fatalError("BF16 cannot be used as the register precision for C.")
    }
    
    // Declare the size of M and N within a register allocation.
    registerM = blockDimensions.M / splits.M
    registerN = blockDimensions.N / splits.N
    
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
    leadingDimensions = ("", "")
    leadingBlockDimensions = (.zero, .zero)
    if transposeState.A {
      leadingDimensions.A = "M"
      leadingBlockDimensions.A = paddedBlockDimensionsA.M
    } else {
      leadingDimensions.A = "K"
      leadingBlockDimensions.A = paddedBlockDimensionsA.K
    }
    if transposeState.B {
      leadingDimensions.B = "K"
      leadingBlockDimensions.B = paddedBlockDimensionsB.K
    } else {
      leadingDimensions.B = "N"
      leadingBlockDimensions.B = paddedBlockDimensionsB.N
    }
    
    // Determine the threadgroup memory allocation.
    do {
      blockBytes = (
        A: paddedBlockDimensionsA.M * paddedBlockDimensionsA.K,
        B: paddedBlockDimensionsB.K * paddedBlockDimensionsB.N,
        C: paddedBlockDimensionsC.M * paddedBlockDimensionsC.N)
      blockBytes.A *= UInt16(memoryPrecisions.A.size)
      blockBytes.B *= UInt16(memoryPrecisions.B.size)
      blockBytes.C *= UInt16(memoryPrecisions.C.size)
    }
    
    source = """

\(createMetalSimdgroupEvent())
\(createMetalSimdgroupMatrixStorage())
using namespace metal;

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

\(createUtilities())

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

struct Arguments {
  bool accumulateC;
};

kernel void gemm(device \(memoryName("A")) *A [[buffer(0)]],
                 device \(memoryName("B")) *B [[buffer(1)]],
                 device \(memoryName("C")) *C [[buffer(2)]],
                 constant Arguments &arguments [[buffer(30)]],
                 threadgroup uchar *threadgroup_block [[threadgroup(0)]],
                 
                 uint3 gid [[threadgroup_position_in_grid]],
                 ushort sidx [[simdgroup_index_in_threadgroup]],
                 ushort lane_id [[thread_index_in_simdgroup]])
{
  auto A_block = (threadgroup \(memoryName("A"))*)(
    threadgroup_block);
  auto B_block = (threadgroup \(memoryName("B"))*)(
    threadgroup_block + \(blockBytes.A));
  
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

  \(createLoadC())
  \(createMultiplyIterations())
  \(createStoreC())
}

"""
    
    // Compile the shader source.
    library = try! device.makeLibrary(source: source, options: nil)
  }
}

extension GEMMKernel {
  func memoryName(_ operand: String) -> String {
    switch operand {
    case "A":
      return memoryPrecisions.A.name
    case "B":
      return memoryPrecisions.B.name
    case "C":
      return memoryPrecisions.C.name
    default:
      fatalError("Unrecognized operand.")
    }
  }
  
  func registerName(_ operand: String) -> String {
    switch operand {
    case "A":
      return registerPrecisions.A.name
    case "B":
      return registerPrecisions.B.name
    case "C":
      return registerPrecisions.C.name
    default:
      fatalError("Unrecognized operand.")
    }
  }
  
  var threadgroupMemoryAllocation: UInt16 {
    max(blockBytes.A + blockBytes.B, blockBytes.C)
  }
}
