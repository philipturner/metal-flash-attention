//
//  AttentionKernel.swift
//  FlashAttention
//
//  Created by Philip Turner on 6/27/24.
//

// Design a set of simple kernels for forward and backward FlashAttention:
// - FP32 (hardcoded data type keyword)
// - 32x32 block, 4 splits (hardcoded block size)
// - all GEMM operands accessed like with standard GEMM + M1
//   - use async copies
//   - transposes are supported
// - no masking, dropout, etc.

struct AttentionKernel {
  // The source code to compile.
  var source: String = ""
  
  // These variables should be 'private', but we need to split the code into
  // multiple files. Swift treats 'private' as a synonym for 'fileprivate'.
  var leadingDimensions: (
    Q: String, K: String, V: String, O: String)
  var leadingBlockDimensions: (
    Q: UInt16, K: UInt16, V: UInt16, O: UInt16)
  var matrixDimensionD: UInt16
  var memoryPrecisions: (
    Q: AttentionOperandPrecision,
    K: AttentionOperandPrecision,
    V: AttentionOperandPrecision,
    O: AttentionOperandPrecision)
  var paddedD: UInt16
  var transposeState: (
    Q: Bool, K: Bool, V: Bool, O: Bool)
  
  // Reads of very large K/V operands may be read in small chunks along 'D',
  // to minimize register pressure. Therefore, there can be a block dimension
  // for D.
  var blockDimensions: (R: UInt16, C: UInt16, D: UInt16)
  
  // The row stride of the intermediate attention matrix.
  var leadingDimensionDerivativeST: UInt32
  
  // If you allocate threadgroup memory after compiling the kernel, the code
  // has higher performance.
  var threadgroupMemoryAllocation: UInt16
  
  // The number of threads per group.
  var threadgroupSize: UInt16
  
  init(descriptor: AttentionDescriptor) {
    guard let matrixDimensions = descriptor.matrixDimensions,
          let memoryPrecisions = descriptor.memoryPrecisions,
          let transposeState = descriptor.transposeState,
          let type = descriptor.type else {
      fatalError("Descriptor was incomplete.")
    }
    self.matrixDimensionD = matrixDimensions.D
    self.memoryPrecisions = memoryPrecisions
    self.transposeState = transposeState
    
    // Inject the contents of the headers.
    source += """
\(createMetalSimdgroupEvent())
\(createMetalSimdgroupMatrixStorage())
using namespace metal;

"""
    
    // Declare the size of the register allocation.
    paddedD = (matrixDimensions.D + 8 - 1) / 8 * 8
    
    // Determine the block dimensions from the transpose state.
    leadingDimensions = ("D", "D", "D", "D")
    leadingBlockDimensions = (paddedD, paddedD, paddedD, paddedD)
    if transposeState.Q {
      leadingDimensions.Q = "R"
      leadingBlockDimensions.Q = 32
    }
    if transposeState.K {
      leadingDimensions.K = "C"
      leadingBlockDimensions.K = 32
    }
    if transposeState.V {
      leadingDimensions.V = "C"
      leadingBlockDimensions.V = 32
    }
    if transposeState.O {
      leadingDimensions.O = "R"
      leadingBlockDimensions.O = 32
    }
    leadingDimensionDerivativeST = matrixDimensions.C + 32 - 1
    leadingDimensionDerivativeST = leadingDimensionDerivativeST / 32 * 32
    
    blockDimensions = (R: 32, C: 32, D: paddedD)
    threadgroupMemoryAllocation = .zero
    threadgroupSize = 128
    
    source += """

// Dimensions of each matrix.
constant uint R [[function_constant(0)]];
constant uint C [[function_constant(1)]];
constant ushort D [[function_constant(2)]];

// Define the memory layout of the matrix block.
constant ushort R_group = 32;
constant ushort C_group = 32;

// Declare the function.
kernel void attention(

"""
    
    // R/C_group * D * sizeof(float)
    threadgroupMemoryAllocation += 32 * paddedD * 4
    
    // Temporary patch, until the new versions of the kernels are finished.
    threadgroupMemoryAllocation *= 2
    
    source += createArguments(type: type)
    source += createSetup(type: type)
    switch type {
    case .forward:
      source += createInnerLoopForward()
      
    case .backwardQuery(let computeDerivativeQ):
      if computeDerivativeQ {
        source += createInnerLoopBackwardQuery()
      }
    case .backwardKeyValue(let computeDerivativeK):
      // R_group * sizeof(float)
      threadgroupMemoryAllocation += 32 * 4
      
      source += createInnerLoopKeyValue(
        computeDerivativeK: computeDerivativeK)
    }
    source += createCleanup(type: type)
    source += """

}

"""
  }
}

extension AttentionKernel {
  func createArguments(type: AttentionKernelType) -> String {
    struct AttentionOperand {
      var precision: GEMMOperandPrecision
      var bufferBinding: Int
    }
    
    // Index the operands available during the forward pass.
    var operandsMap: [String: AttentionOperand] = [:]
    operandsMap["Q"] = AttentionOperand(
      precision: memoryPrecisions.Q.forwardPrecision, bufferBinding: 0)
    operandsMap["K"] = AttentionOperand(
      precision: memoryPrecisions.K.forwardPrecision, bufferBinding: 1)
    operandsMap["V"] = AttentionOperand(
      precision: memoryPrecisions.V.forwardPrecision, bufferBinding: 2)
    operandsMap["O"] = AttentionOperand(
      precision: memoryPrecisions.O.forwardPrecision, bufferBinding: 3)
    operandsMap["L_terms"] = AttentionOperand(
      precision: memoryPrecisions.O.forwardPrecision, bufferBinding: 4)
    
    // Index the operands available during the backward pass.
    operandsMap["dO"] = AttentionOperand(
      precision: memoryPrecisions.O.backwardPrecision, bufferBinding: 5)
    operandsMap["D_terms"] = AttentionOperand(
      precision: memoryPrecisions.O.backwardPrecision, bufferBinding: 6)
    operandsMap["dV"] = AttentionOperand(
      precision: memoryPrecisions.V.backwardPrecision, bufferBinding: 7)
    operandsMap["dST"] = AttentionOperand(
      // The default kernel doesn't support writing the attention matrix to
      // memory. The purpose of dS is to increase performance when possible. If
      // users wanted to set dS to FP32 for correctness, that would defeat the
      // purpose. In addition, dS serves as a temporary allocation. Its
      // contents should not be visible to any code that would measure
      // numerical correctness.
      //
      // This is an intermediate allocation, managed internally by the MFA
      // backend. We can impose constraints on it that wouldn't typically be
      // feasible. For example, we can force the row stride to be divisible by
      // the block size (32). This simplifies the code; we don't need to run
      // async copies to safeguard against corrupted memory accesses.
      //
      // If the matrix rows are noncontiguous, we must modify the in-tree
      // GEMM kernel to support custom leading dimensions. This can be
      // something modified explicitly by the user - an option to override the
      // default leading dimension.
      precision: AttentionOperandPrecision.mixed.backwardPrecision,
      bufferBinding: 8)
    operandsMap["dK"] = AttentionOperand(
      precision: memoryPrecisions.K.backwardPrecision, bufferBinding: 8)
    operandsMap["dQ"] = AttentionOperand(
      precision: memoryPrecisions.Q.backwardPrecision, bufferBinding: 9)
    
    // Select the operands used by this variant.
    var operandKeys: [String]
    switch type {
    case .forward(let computeL):
      operandKeys = [
        "Q", "K", "V", "O"
      ]
      if computeL {
        operandKeys.append("L_terms")
      }
    case .backwardQuery(let computeDerivativeQ):
      if computeDerivativeQ {
        operandKeys = [
          "Q", "K", "V", "O",
          "L_terms", "dO", "D_terms", "dQ"
        ]
      } else {
        operandKeys = [
          "O", "dO", "D_terms"
        ]
      }
    case .backwardKeyValue(let computeDerivativeK):
      operandKeys = [
        "Q", "K", "V",
        "L_terms", "dO", "D_terms", "dV"
      ]
      if computeDerivativeK {
        operandKeys.append("dK")
      } else {
        operandKeys.append("dST")
      }
    }
    
    // Collect the operands into a single string.
    var output: String = ""
    for key in operandKeys {
      let operand = operandsMap[key]!
      
      var line = "  "
      line += "device "
      line += operand.precision.name + " "
      line += "*" + key + " "
      line += "[[buffer(\(operand.bufferBinding))]]"
      line += ",\n"
      output += line
    }
    
    // Add the arguments that define the thread's position.
    output += """
  
  threadgroup uchar *threadgroup_block [[threadgroup(0)]],
  
  uint gid [[threadgroup_position_in_grid]],
  ushort sidx [[simdgroup_index_in_threadgroup]],
  ushort lane_id [[thread_index_in_simdgroup]]
) {
  ushort2 morton_offset = morton_order(lane_id);

"""
    
    // The thread's array slot in the row or column dimension (whichever the
    // kernel is parallelized over). Used for indexing into 1D arrays.
    switch type {
    case .forward, .backwardQuery:
      output += """

  uint linear_array_slot = gid * R_group + sidx * 8 + morton_offset.y;

"""
    default:
      break
    }
    
    return output
  }
}

// MARK: - Setup and Cleanup Specification

// SIMD Matrix Storage
//   register allocation per thread
//    (scalar footprint) * (2 * D / 8) bytes
// -> (scalar footprint) * (D / 4) bytes
//
// Forward
//   cache Q, O, m, l
//     FP32: 8 * (D / 4) + 8 bytes
//     FP16: 6 * (D / 4) + 8 bytes
//
// Backward Query (true)
//   cache dQ, dO, L[i], D[i]
//     FP32: 8 * (D / 4) + 8 bytes
//     FP16: 6 * (D / 4) + 8 bytes
//   cache Q
//     FP32: 12 * (D / 4) + 8 bytes
//     FP16:  8 * (D / 4) + 8 bytes
//
// Backward Key-Value (true)
//   cache dK, dV
//     FP32: 8 * (D / 4) bytes
//     FP16: 8 * (D / 4) bytes
//   cache K, V
//     FP32: 16 * (D / 4) bytes
//     FP16: 12 * (D / 4) bytes
//
// Backward Key-Value (false)
//   cache dV
//     FP32: 4 * (D / 4) bytes
//     FP16: 4 * (D / 4) bytes
//   cache K, V
//     FP32: 12 * (D / 4) bytes
//     FP16:  8 * (D / 4) bytes
//
// Need code for:
//   prefetching and loading 2D matrices (with async copy)
//   loading 1D operands directly from device (with a single conditional)
//     returning early when a SIMD is out of bounds
//   initializing accumulators
//
// This code should be possible to repurpose during the prefetches for
// matrix multiplication. The next part that logically follows is the
// store operations and the tear-down procedure.
//
// Instead of a monolithic function for "set(U/u)p" and "tear(D/d)own",
// it might be better to form a programmable API. Specify the operands,
// which order they appear (to ease prefetching). Wrap each generic
// or operand-specific procedure into a modular building block.

extension AttentionKernel {
  func createSetup(type: AttentionKernelType) -> String {
    var output: String = ""
    
    // Loading everything that could possibly be loaded, for now.
    switch type {
    case .forward:
      // Q, O
      var accessDesc = AttentionHBMAccessDescriptor()
      accessDesc.index = "gid * R_group"
      accessDesc.leadingBlockDimension = leadingBlockDimensions.Q
      accessDesc.leadingDimension = leadingDimensions.Q
      accessDesc.name = "Q"
      accessDesc.threadgroupAddress = "threadgroup_block"
      accessDesc.transposeState = transposeState.Q
      
      // output += prefetchRows(descriptor: accessDesc)
      output += zeroInitializeAccumulator(name: "O")
      output += threadgroupBarrier()
      // output += load(descriptor: accessDesc)
      
      // m, l
      output += """

  float m = -numeric_limits<float>::max();
  float l = numeric_limits<float>::denorm_min();

"""
      
    case .backwardQuery(let computeDerivativeQ):
      // O, dO, D[i]
      do {
        var accessDesc = AttentionHBMAccessDescriptor()
        accessDesc.index = "gid * R_group"
        accessDesc.leadingBlockDimension = leadingBlockDimensions.O
        accessDesc.leadingDimension = leadingDimensions.O
        accessDesc.threadgroupAddress = "threadgroup_block"
        accessDesc.transposeState = transposeState.O
        
        accessDesc.name = "O"
        output += prefetchRows(descriptor: accessDesc)
        output += threadgroupBarrier()
        output += load(descriptor: accessDesc)
        
        accessDesc.name = "dO"
        output += prefetchRows(descriptor: accessDesc)
        output += threadgroupBarrier()
        output += load(descriptor: accessDesc)
        
        output += """

  float D_term = 0;
#pragma clang loop unroll(full)
  for (ushort d = 0; d < \(paddedD); d += 8) {
    float2 O_value = *(O_sram[d / 8].thread_elements());
    float2 dO_value = *(dO_sram[d / 8].thread_elements());
    D_term += O_value[0] * dO_value[0];
    D_term += O_value[1] * dO_value[1];
  }
  D_term += simd_shuffle_xor(D_term, 1);
  D_term += simd_shuffle_xor(D_term, 8);
  D_term *= 1 / sqrt(float(D));

"""
      }
      
      // Q, dQ, L[i]
      if computeDerivativeQ {
        var accessDesc = AttentionHBMAccessDescriptor()
        accessDesc.index = "gid * R_group"
        accessDesc.leadingBlockDimension = leadingBlockDimensions.Q
        accessDesc.leadingDimension = leadingDimensions.Q
        accessDesc.name = "Q"
        accessDesc.threadgroupAddress = "threadgroup_block"
        accessDesc.transposeState = transposeState.Q
        
        output += prefetchRows(descriptor: accessDesc)
        output += """

  float L_term = L_terms[linear_array_slot];

"""
        output += zeroInitializeAccumulator(name: "dQ")
        output += threadgroupBarrier()
        output += load(descriptor: accessDesc)
      }
      
    case .backwardKeyValue(let computeDerivativeK):
      // dK, K
      do {
        var accessDesc = AttentionHBMAccessDescriptor()
        accessDesc.index = "gid * C_group"
        accessDesc.leadingBlockDimension = leadingBlockDimensions.K
        accessDesc.leadingDimension = leadingDimensions.K
        accessDesc.name = "K"
        accessDesc.threadgroupAddress = "threadgroup_block"
        accessDesc.transposeState = transposeState.K
        
        output += prefetchColumns(descriptor: accessDesc)
        if computeDerivativeK {
          output += zeroInitializeAccumulator(name: "dK")
        }
        output += threadgroupBarrier()
        output += load(descriptor: accessDesc)
      }
      
      // dV, V
      do {
        var accessDesc = AttentionHBMAccessDescriptor()
        accessDesc.index = "gid * C_group"
        accessDesc.leadingBlockDimension = leadingBlockDimensions.V
        accessDesc.leadingDimension = leadingDimensions.V
        accessDesc.name = "V"
        accessDesc.threadgroupAddress = "threadgroup_block"
        accessDesc.transposeState = transposeState.V
        
        output += prefetchColumns(descriptor: accessDesc)
        output += zeroInitializeAccumulator(name: "dV")
        output += threadgroupBarrier()
        output += load(descriptor: accessDesc)
      }
    }
    
    return output
  }
  
  func createCleanup(type: AttentionKernelType) -> String {
    var output: String = ""
    
    switch type {
    case .forward(let computeL):
      // O
      var accessDesc = AttentionHBMAccessDescriptor()
      accessDesc.index = "gid * R_group"
      accessDesc.leadingBlockDimension = leadingBlockDimensions.O
      accessDesc.leadingDimension = leadingDimensions.O
      accessDesc.name = "O"
      accessDesc.threadgroupAddress = "threadgroup_block"
      accessDesc.transposeState = transposeState.O
      
      output += threadgroupBarrier()
      output += store(descriptor: accessDesc)
      output += threadgroupBarrier()
      output += commitRows(descriptor: accessDesc)
      
      // L[i]
      if computeL {
        output += """

  // Premultiplied by M_LOG2E_F.
  float L_term = m + fast::log2(l);
  L_terms[linear_array_slot] = L_term;

"""
      }
      
    case .backwardQuery(let computeDerivativeQ):
      // dQ
      if computeDerivativeQ {
        var accessDesc = AttentionHBMAccessDescriptor()
        accessDesc.index = "gid * R_group"
        accessDesc.leadingBlockDimension = leadingBlockDimensions.Q
        accessDesc.leadingDimension = leadingDimensions.Q
        accessDesc.name = "dQ"
        accessDesc.threadgroupAddress = "threadgroup_block"
        accessDesc.transposeState = transposeState.Q
        
        output += threadgroupBarrier()
        output += store(descriptor: accessDesc)
        output += threadgroupBarrier()
        output += commitRows(descriptor: accessDesc)
      }
      
      // D[i]
      output += """

  D_terms[linear_array_slot] = D_term;

"""
      
    case .backwardKeyValue(let computeDerivativeK):
      // dK
      if computeDerivativeK {
        var accessDesc = AttentionHBMAccessDescriptor()
        accessDesc.index = "gid * C_group"
        accessDesc.leadingBlockDimension = leadingBlockDimensions.K
        accessDesc.leadingDimension = leadingDimensions.K
        accessDesc.name = "dK"
        accessDesc.threadgroupAddress = "threadgroup_block"
        accessDesc.transposeState = transposeState.K
        
        
        output += threadgroupBarrier()
        output += store(descriptor: accessDesc)
        output += threadgroupBarrier()
        output += commitColumns(descriptor: accessDesc)
      }
      
      // dV
      do {
        var accessDesc = AttentionHBMAccessDescriptor()
        accessDesc.index = "gid * C_group"
        accessDesc.leadingBlockDimension = leadingBlockDimensions.V
        accessDesc.leadingDimension = leadingDimensions.V
        accessDesc.name = "dV"
        accessDesc.threadgroupAddress = "threadgroup_block"
        accessDesc.transposeState = transposeState.V
          
        output += threadgroupBarrier()
        output += store(descriptor: accessDesc)
        output += threadgroupBarrier()
        output += commitColumns(descriptor: accessDesc)
      }
      
    }
    
    return output
  }
  
  func zeroInitializeAccumulator(name: String) -> String {
    return """

  simdgroup_matrix_storage<float> \(name)_sram[\(paddedD / 8)];
#pragma clang loop unroll(full)
  for (ushort d = 0; d < \(paddedD); d += 8) {
    \(name)_sram[d / 8] = simdgroup_matrix_storage<float>(0);
  }

"""
  }
  
  // A threadgroup barrier, formatted to match the correct indentation.
  func threadgroupBarrier() -> String {
    return """

  threadgroup_barrier(mem_flags::mem_threadgroup);

"""
  }
}

// MARK: - Load

struct AttentionHBMAccessDescriptor {
  var index: String?
  var leadingBlockDimension: UInt16?
  var leadingDimension: String?
  var name: String?
  var threadgroupAddress: String?
  var transposeState: Bool?
}

extension AttentionKernel {
  // Cache something during a pass that parallelizes over rows.
  func prefetchRows(descriptor: AttentionHBMAccessDescriptor) -> String {
    guard let index = descriptor.index,
          let leadingBlockDimension = descriptor.leadingBlockDimension,
          let leadingDimension = descriptor.leadingDimension,
          let name = descriptor.name,
          let threadgroupAddress = descriptor.threadgroupAddress,
          let transposeState = descriptor.transposeState else {
      fatalError("Descriptor was incomplete.")
    }
    
    return """

  if (sidx == 0) {
    uint2 device_origin(0, \(index));
    auto src = simdgroup_matrix_storage<float>::apply_offset(
      \(name), \(leadingDimension), device_origin, \(transposeState));
    auto dst = (threadgroup float*)(\(threadgroupAddress));
    
    ushort R_tile_dimension = min(uint(R_group), R - \(index));
    ushort2 tile_src(D, R_tile_dimension);
    
    // TODO: Fix the kernels, so you don't have to zero-pad this.
    ushort2 tile_dst(\(paddedD), R_group);
    
    simdgroup_event event;
    event.async_copy(
      dst, \(leadingBlockDimension), tile_dst,
      src, \(leadingDimension), tile_src, \(transposeState));
    simdgroup_event::wait(1, &event);
  }

"""
  }
  
  // Cache something during a pass that parallelizes over columns.
  func prefetchColumns(descriptor: AttentionHBMAccessDescriptor) -> String {
    guard let index = descriptor.index,
          let leadingBlockDimension = descriptor.leadingBlockDimension,
          let leadingDimension = descriptor.leadingDimension,
          let name = descriptor.name,
          let threadgroupAddress = descriptor.threadgroupAddress,
          let transposeState = descriptor.transposeState else {
      fatalError("Descriptor was incomplete.")
    }
    
    return """

  if (sidx == 0) {
    uint2 device_origin(0, \(index));
    auto src = simdgroup_matrix_storage<float>::apply_offset(
      \(name), \(leadingDimension), device_origin, \(transposeState));
    auto dst = (threadgroup float*)(\(threadgroupAddress));
    
    ushort C_tile_dimension = min(uint(C_group), C - \(index));
    ushort2 tile_src(D, C_tile_dimension);
    
    // TODO: Fix the kernels, so you don't have to zero-pad this.
    ushort2 tile_dst(\(paddedD), C_group);
    
    simdgroup_event event;
    event.async_copy(
      dst, \(leadingBlockDimension), tile_dst,
      src, \(leadingDimension), tile_src, \(transposeState));
    simdgroup_event::wait(1, &event);
  }

"""
  }
  
  func load(descriptor: AttentionHBMAccessDescriptor) -> String {
    guard let leadingBlockDimension = descriptor.leadingBlockDimension,
          let name = descriptor.name,
          let threadgroupAddress = descriptor.threadgroupAddress,
          let transposeState = descriptor.transposeState else {
      fatalError("Descriptor was incomplete.")
    }
    
    return """

  simdgroup_matrix_storage<float> \(name)_sram[\(paddedD / 8)];
  {
    ushort2 threadgroup_origin = ushort2(0, sidx * 8) + morton_offset;
    auto src = (threadgroup float*)(\(threadgroupAddress));
    src = simdgroup_matrix_storage<float>::apply_offset(
      src, \(leadingBlockDimension), threadgroup_origin, \(transposeState));
    
#pragma clang loop unroll(full)
    for (ushort d = 0; d < \(paddedD); d += 8) {
      ushort2 thread_origin(d, 0);
      \(name)_sram[d / 8].load(
        src, \(leadingBlockDimension), thread_origin, \(transposeState));
    }
  }

"""
  }
}

// MARK: - Store

extension AttentionKernel {
  func store(descriptor: AttentionHBMAccessDescriptor) -> String {
    guard let leadingBlockDimension = descriptor.leadingBlockDimension,
          let name = descriptor.name,
          let threadgroupAddress = descriptor.threadgroupAddress,
          let transposeState = descriptor.transposeState else {
      fatalError("Descriptor was incomplete.")
    }
    
    return """

  {
    ushort2 threadgroup_origin = ushort2(0, sidx * 8) + morton_offset;
    auto dst = (threadgroup float*)(\(threadgroupAddress));
    dst = simdgroup_matrix_storage<float>::apply_offset(
      dst, \(leadingBlockDimension), threadgroup_origin, \(transposeState));
    
#pragma clang loop unroll(full)
    for (ushort d = 0; d < \(paddedD); d += 8) {
      ushort2 thread_origin(d, 0);
      \(name)_sram[d / 8].store(
        dst, \(leadingBlockDimension), thread_origin, \(transposeState));
    }
  }

"""
  }
  
  // Finish something during a pass that parallelizes over rows.
  func commitRows(descriptor: AttentionHBMAccessDescriptor) -> String {
    guard let index = descriptor.index,
          let leadingBlockDimension = descriptor.leadingBlockDimension,
          let leadingDimension = descriptor.leadingDimension,
          let name = descriptor.name,
          let threadgroupAddress = descriptor.threadgroupAddress,
          let transposeState = descriptor.transposeState else {
      fatalError("Descriptor was incomplete.")
    }
    
    return """

  if (sidx == 0) {
    uint2 device_origin(0, \(index));
    auto src = (threadgroup float*)(\(threadgroupAddress));
    auto dst = simdgroup_matrix_storage<float>::apply_offset(
      \(name), \(leadingDimension), device_origin, \(transposeState));
   
    ushort R_tile_dimension = min(uint(R_group), R - \(index));
    ushort2 tile_src(D, R_tile_dimension);
    ushort2 tile_dst(D, R_tile_dimension);
    
    simdgroup_event event;
    event.async_copy(
      dst, \(leadingDimension), tile_dst,
      src, \(leadingBlockDimension), tile_src, \(transposeState));
    simdgroup_event::wait(1, &event);
  }

"""
  }
  
  // Finish something during a pass that parallelizes over columns.
  func commitColumns(descriptor: AttentionHBMAccessDescriptor) -> String {
    guard let index = descriptor.index,
          let leadingBlockDimension = descriptor.leadingBlockDimension,
          let leadingDimension = descriptor.leadingDimension,
          let name = descriptor.name,
          let threadgroupAddress = descriptor.threadgroupAddress,
          let transposeState = descriptor.transposeState else {
      fatalError("Descriptor was incomplete.")
    }
    
    return """

  if (sidx == 0) {
    uint2 device_origin(0, \(index));
    auto src = (threadgroup float*)(\(threadgroupAddress));
    auto dst = simdgroup_matrix_storage<float>::apply_offset(
      \(name), \(leadingDimension), device_origin, \(transposeState));
   
    ushort C_tile_dimension = min(uint(C_group), C - \(index));
    ushort2 tile_src(D, C_tile_dimension);
    ushort2 tile_dst(D, C_tile_dimension);
    
    simdgroup_event event;
    event.async_copy(
      dst, \(leadingDimension), tile_dst,
      src, \(leadingBlockDimension), tile_src, \(transposeState));
    simdgroup_event::wait(1, &event);
  }

"""
  }
}
