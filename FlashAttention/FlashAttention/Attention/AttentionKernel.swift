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
  
  private var leadingDimensions: (
    Q: String, K: String, V: String, O: String)
  private var leadingBlockDimensions: (
    Q: UInt16, K: UInt16, V: UInt16, O: UInt16)
  private var matrixDimensionD: UInt16
  private var memoryPrecisions: (
    Q: AttentionOperandPrecision,
    K: AttentionOperandPrecision,
    V: AttentionOperandPrecision,
    O: AttentionOperandPrecision)
  private var paddedD: UInt16
  private var transposeState: (
    Q: Bool, K: Bool, V: Bool, O: Bool)
  
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
    
    source += """

// Dimensions of each matrix.
constant uint R [[function_constant(0)]];
constant uint C [[function_constant(1)]];
constant uint D [[function_constant(2)]];

// Define the memory layout of the matrix block.
constant ushort R_group = 32;
constant ushort C_group = 32;

// Declare the function.
kernel void attention(

"""
    
    source += createArguments(type: type)
    source += createSetup(type: type)
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
    operandsMap["dS"] = AttentionOperand(
      // The default kernel doesn't support writing the attention matrix to
      // memory. The purpose of dS is to increase performance when possible. If
      // users wanted to set dS to FP32 for correctness, that would defeat the
      // purpose. In addition, dS serves as a temporary allocation. Its
      // contents should not be visible to any code that would measure
      // numerical correctness.
      //
      // TODO: Find a workable way to set 'L', 'D_terms', and 'dS' terms to
      // reasonable default values. While allowing users to override them for
      // debugging purposes.
      //
      // Perhaps once a distinct 'AttentionKernelDescriptor" is coded.
      //
      // Idea: "attentionMatrixPrecision". This property equals both the
      // register type and the type when paged to memory.
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
        operandKeys.append("dS")
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
    case .backwardKeyValue:
      output += """

  uint linear_array_slot = gid * C_group + sidx * 8 + morton_offset.y;

"""
    }
    
    return output
  }
}

// MARK: - Setup and Cleanup Specification

// Forward
//   cache Q, O, m, l
//     FP32: 8 * 2 * D + 8 bytes
//     FP16: 6 * 2 * D + 8 bytes
//
// Backward Query (true)
//   cache dQ, dO, L[i], D[i]
//     FP32: 8 * 2 * D + 8 bytes
//     FP16: 6 * 2 * D + 8 bytes
//   cache Q
//     FP32: 12 * 2 * D + 8 bytes
//     FP16:  8 * 2 * D + 8 bytes
//
// Backward Key-Value (true)
//   cache dK, dV
//     FP32: 8 * 2 * D bytes
//     FP16: 8 * 2 * D bytes
//   cache K, V
//     FP32: 16 * 2 * D bytes
//     FP16: 12 * 2 * D bytes
//
// Backward Key-Value (false)
//   cache dV
//     FP32: 4 * 2 * D bytes
//     FP16: 4 * 2 * D bytes
//   cache K, V
//     FP32: 12 * 2 * D bytes
//     FP16:  8 * 2 * D bytes
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
      
      output += prefetchRows(descriptor: accessDesc)
      output += zeroInitializeAccumulator(name: "O")
      output += threadgroupBarrier()
      output += load(descriptor: accessDesc)
      
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
  float L_term = fast::log2(l) * M_LOG2E_F + m;
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

// Load a chunk of the operand into registers.
//
// Returns:
// - a string for loading from device -> threadgroup
// - a string for loading from threadgroup -> thread
//
// The second string may be deferred until a much later time, after a
// threadgroup barrier. Ideally, one would perform other work while waiting
// on the 'device -> threadgroup' copy to happen asynchronously.
//
// THIS IS NOT MEANT TO BE USED INSIDE THE INNER GEMM LOOP

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
    ushort2 tile_dst(\(paddedD), R_tile_dimension);
    
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
    ushort2 tile_dst(\(paddedD), C_tile_dimension);
    
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
  }

"""
  }
}

// MARK: - Inner Loop

// Forward
//   for c in 0..<C {
//     load K[c]
//     S = Q * K^T
//     (m, l, P) = softmax(m, l, S * scaleFactor)
//     O *= correction
//     load V[c]
//     O += P * V
//   }
//   O /= l
//
// Backward Query (true)
//   for c in 0..<C {
//     load K[c]
//     S = Q * K^T
//     P = exp(S - L)
//     load V[c]
//     dP = dO * V
//     dS = P * (dP - D) * scaleFactor
//     load K[c]
//     dQ += dS * K
//   }
//
// Backward Key-Value (false)
//   for r in 0..<R {
//     load Q[r]
//     load L[r]
//     S^T = K * Q^T
//     P^T = exp(S^T - L)
//     load dO[r]
//     dV += P^T * dO
//     load V[r]
//     load D[r]
//     dP = dO * V^T
//     dS = P * (dP - D) * scaleFactor
//     store dS[r][c]
//   }
//
// Backward Key-Value (true)
//   for r in 0..<R {
//     load Q[r]
//     load L[r]
//     S^T = K * Q^T
//     P^T = exp(S^T - L)
//     load dO[r]
//     dV += P^T * dO
//     load V[r]
//     load D[r]
//     dP = dO * V^T
//     dS = P * (dP - D) * scaleFactor
//     load Q[r]
//     dK += dS^T * Q
//   }

extension AttentionKernel {
  // Start by writing the inner loop for the forward kernel. This may reveal
  // high-level abstractions that can be shared with the remaining variants.
  func createInnerLoopForward() -> String {
    var prefetchK: String
    do {
      var accessDesc = AttentionHBMAccessDescriptor()
      accessDesc.index = "c"
      accessDesc.leadingBlockDimension = leadingBlockDimensions.K
      accessDesc.leadingDimension = leadingDimensions.K
      accessDesc.name = "K"
      accessDesc.threadgroupAddress = "threadgroup_block"
      accessDesc.transposeState = transposeState.K
      prefetchK = prefetchRows(descriptor: accessDesc)
    }
    
    var prefetchV: String
    do {
      var accessDesc = AttentionHBMAccessDescriptor()
      accessDesc.index = "c"
      accessDesc.leadingBlockDimension = leadingBlockDimensions.V
      accessDesc.leadingDimension = leadingDimensions.V
      accessDesc.name = "V"
      accessDesc.threadgroupAddress = "threadgroup_block"
      accessDesc.transposeState = transposeState.V
      prefetchV = prefetchRows(descriptor: accessDesc)
    }
    
    var scaleFactor = Float(1.44269504089) // M_LOG2E_F
    scaleFactor /= Float(matrixDimensionD).squareRoot()
    
    return """
  
  // Iterate over the columns.
  for (uint c = 0; c < C; c += 32) {
    // load K[c]
    threadgroup_barrier(mem_flags::mem_threadgroup);
    \(prefetchK)
    auto KT_block = (threadgroup float*)(threadgroup_block);
    KT_block = simdgroup_matrix_storage<float>::apply_offset(
      KT_block, \(leadingBlockDimensions.K), morton_offset,
      \(!transposeState.K));
    
    // S = Q * K^T
    threadgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_matrix_storage<float> S_sram[32 / 8];
#pragma clang loop unroll(full)
    for (ushort d = 0; d < \(paddedD); d += 8) {
#pragma clang loop unroll(full)
      for (ushort c = 0; c < 32; c += 8) {
        ushort2 origin(c, d);
        simdgroup_matrix_storage<float> KT;
        KT.load(
          KT_block, \(leadingBlockDimensions.K), origin, \(!transposeState.K));
        S_sram[c / 8]
          .multiply(Q_sram[d / 8], KT, d > 0);
      }
    }

    // Prevent the zero padding from changing the values of 'm' and 'l'.
    if ((C % 32 != 0) && (c + 32 > C)) {
      const ushort remainder32 = C - uint(C % 32);
      const ushort remainder8 = C - uint(C % 8);
      const ushort remainder32_floor = remainder32 - ushort(remainder32 % 8);
      
#pragma clang loop unroll(full)
      for (ushort index = 0; index < 2; ++index) {
        if (offset_in_simd.x + index >= remainder32 - remainder32_floor) {
          auto S_elements = S_sram[remainder32_floor / 8].thread_elements();
          *(S_elements)[index] = -numeric_limits<float>::max();
        }
      }
#pragma clang loop unroll(full)
      for (ushort c = remainder32 - remainder32_floor; c < 32; c += 8) {
        auto S_elements = S_sram[c / 8].thread_elements();
        *(S_elements) = -numeric_limits<float>::max();
      }
    }
    
    // update 'm'
    float2 m_new_accumulator;
#pragma clang loop unroll(full)
    for (ushort c = 0; c < 32; c += 8) {
      auto S_elements = S_sram[c / 8].thread_elements();
      if (c == 0) {
        m_new_accumulator = *S_elements;
      } else {
        m_new_accumulator = max(m_new_accumulator, *S_elements);
      }
    }
    float m_new = max(m_new_accumulator[0], m_new_accumulator[1]);
    m_new = max(m_new, simd_shuffle_xor(m_new, 1));
    m_new = max(m_new, simd_shuffle_xor(m_new, 8));
    m_new *= \(scaleFactor);

    // update the previous value of 'O'
    float correction = 1;
    if (m_new > m) {
      correction = fast::exp2(m - m_new);
#pragma clang loop unroll(full)
      for (ushort d = 0; d < \(paddedD); d += 8) {
        *(O_sram[d / 8].thread_elements()) *= correction;
      }
      m = m_new;
    }

    // P = softmax(S * scaleFactor)
    simdgroup_matrix_storage<float> P_sram[32 / 8];
#pragma clang loop unroll(full)
    for (ushort c = 0; c < 32; c += 8) {
      float2 S_elements = float2(*(S_sram[c / 8].thread_elements()));
      float2 P_elements = fast::exp2(S_elements * \(scaleFactor) - m);
      *(P_sram[c / 8].thread_elements()) = P_elements;
    }

    // update 'l'
    float2 l_new_accumulator;
#pragma clang loop unroll(full)
    for (ushort c = 0; c < 32; c += 8) {
      auto P_elements = P_sram[c / 8].thread_elements();
      if (c == 0) {
        l_new_accumulator = *P_elements;
      } else {
        l_new_accumulator += *P_elements;
      }
    }
    float l_new = l_new_accumulator[0] + l_new_accumulator[1];
    l_new += simd_shuffle_xor(l_new, 1);
    l_new += simd_shuffle_xor(l_new, 8);
    l = l * correction + l_new;
    
    // load V[c]
    threadgroup_barrier(mem_flags::mem_threadgroup);
    \(prefetchV)
    auto V_block = (threadgroup float*)(threadgroup_block);
    V_block = simdgroup_matrix_storage<float>::apply_offset(
      V_block, \(leadingBlockDimensions.V), morton_offset, \(transposeState.V));
    
    // O += P * V
    threadgroup_barrier(mem_flags::mem_threadgroup);
#pragma clang loop unroll(full)
    for (ushort c = 0; c < 32; c += 8) {
#pragma clang loop unroll(full)
      for (ushort d = 0; d < \(paddedD); d += 8) {
        ushort2 origin(d, c);
        simdgroup_matrix_storage<float> V;
        V.load(
          V_block, \(leadingBlockDimensions.V), origin, \(transposeState.V));
        O_sram[d / 8]
          .multiply(P_sram[c / 8], V, true);
      }
    }
  }
  
  // O /= l
  float l_reciprocal = 1 / l;
#pragma clang loop unroll(full)
  for (ushort d = 0; d < \(paddedD); d += 8) {
    *(O_sram[d / 8].thread_elements()) *= l_reciprocal;
  }
  
"""
  }
}
