//
//  AttentionKernel+Arguments.swift
//  FlashAttention
//
//  Created by Philip Turner on 7/22/24.
//

// Operations that store data to main memory.

// MARK: - HBM Access

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

struct AttentionHBMAccessDescriptor {
  /// Source or destination of a 32 x D block.
  var name: String?
  
  var transposeState: Bool?
  var leadingDimension: String?
  
  var matrixDimension: String?
  var matrixOffset: String?
}

extension AttentionKernel {
  // Does not have any delay between prefetching something and writing it to
  // registers. This may cause a regression from the old MFA; investigate M1
  // performance at a later date.
  func load(descriptor: AttentionHBMAccessDescriptor) -> String {
    guard let name = descriptor.name,
          let transposeState = descriptor.transposeState,
          let leadingDimension = descriptor.leadingDimension,
          let matrixDimension = descriptor.matrixDimension,
          let matrixOffset = descriptor.matrixOffset else {
      fatalError("Descriptor was incomplete.")
    }
    
    // 32 x 64 allocation in threadgroup memory
    // leading dimension = transposeState ? 32 : 64
    let leadingBlockDimension = transposeState ? UInt16(32) : UInt16(64)
    
    let loopBody = """

ushort2 origin(d, 0);
\(name)_sram[(d_outer + d) / 8].load(
  \(name)_block, \(leadingBlockDimension), origin, \(transposeState));

"""
    
    return """

simdgroup_matrix_storage<float> \(name)_sram[\(paddedD / 8)];
{
  uint M_offset = \(matrixOffset);
  ushort M_src_dimension = min(uint(32), \(matrixDimension) - M_offset);
  
  // Find where the \(name) data will be read from.
  ushort2 \(name)_block_offset(morton_offset.x, morton_offset.y + sidx * 8);
  auto \(name)_block = (threadgroup float*)(threadgroup_block);
  \(name)_block = simdgroup_matrix_storage<float>::apply_offset(
    \(name)_block, \(leadingBlockDimension),
    \(name)_block_offset, \(transposeState));
  
  // Outer loop over D.
#pragma clang loop unroll(full)
  for (ushort d_outer = 0; d_outer < D; d_outer += 64) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    if (sidx == 0) {
      ushort D_src_dimension = min(ushort(64), ushort(D - d_outer));
      ushort D_dst_dimension = min(ushort(64), ushort(\(paddedD) - d_outer));
      
      uint2 \(name)_offset(d_outer, M_offset);
      auto src = simdgroup_matrix_storage<float>::apply_offset(
        \(name), \(leadingDimension), \(name)_offset, \(transposeState));
      auto dst = (threadgroup float*)(threadgroup_block);
     
      ushort2 tile_src(D_src_dimension, M_src_dimension);
      ushort2 tile_dst(D_dst_dimension, M_src_dimension);
      
      simdgroup_event event;
      event.async_copy(
        dst, \(leadingBlockDimension), tile_dst,
        src, \(leadingDimension), tile_src, \(transposeState));
      simdgroup_event::wait(1, &event);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Iterate over the head dimension.
    if (\(paddedD) - d_outer >= 64) {
#pragma clang loop unroll(full)
      for (ushort d = 0; d < 64; d += 8) {
        \(loopBody)
      }
    } else {
#pragma clang loop unroll(full)
      for (ushort d = 0; d < \(paddedD) % 64; d += 8) {
        \(loopBody)
      }
    }
  }
}

"""
  }
  
  func store(descriptor: AttentionHBMAccessDescriptor) -> String {
    guard let name = descriptor.name,
          let transposeState = descriptor.transposeState,
          let leadingDimension = descriptor.leadingDimension,
          let matrixDimension = descriptor.matrixDimension,
          let matrixOffset = descriptor.matrixOffset else {
      fatalError("Descriptor was incomplete.")
    }
    
    // 32 x 64 allocation in threadgroup memory
    // leading dimension = transposeState ? 32 : 64
    let leadingBlockDimension = transposeState ? UInt16(32) : UInt16(64)
    
    let loopBody = """

ushort2 origin(d, 0);
\(name)_sram[(d_outer + d) / 8].store(
  \(name)_block, \(leadingBlockDimension), origin, \(transposeState));

"""
    
    return """

{
  // Find where the \(name) data will be written to.
  ushort2 \(name)_block_offset(morton_offset.x, morton_offset.y + sidx * 8);
  auto \(name)_block = (threadgroup float*)(threadgroup_block);
  \(name)_block = simdgroup_matrix_storage<float>::apply_offset(
    \(name)_block, \(leadingBlockDimension),
    \(name)_block_offset, \(transposeState));
  
  // Outer loop over D.
#pragma clang loop unroll(full)
  for (ushort d_outer = 0; d_outer < D; d_outer += 64) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Iterate over the head dimension.
    if (\(paddedD) - d_outer >= 64) {
#pragma clang loop unroll(full)
      for (ushort d = 0; d < 64; d += 8) {
        \(loopBody)
      }
    } else {
#pragma clang loop unroll(full)
      for (ushort d = 0; d < \(paddedD) % 64; d += 8) {
        \(loopBody)
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    if (sidx == 0) {
      uint2 \(name)_offset(d_outer, \(matrixOffset));
      auto src = (threadgroup float*)(threadgroup_block);
      auto dst = simdgroup_matrix_storage<float>::apply_offset(
        \(name), \(leadingDimension), \(name)_offset, \(transposeState));
     
      ushort D_dimension = min(ushort(64), ushort(D - d_outer));
      ushort RC_dimension = min(
        uint(32), \(matrixDimension) - \(matrixOffset));
      ushort2 tile_src(D_dimension, RC_dimension);
      ushort2 tile_dst(D_dimension, RC_dimension);
      
      simdgroup_event event;
      event.async_copy(
        dst, \(leadingDimension), tile_dst,
        src, \(leadingBlockDimension), tile_src, \(transposeState));
      simdgroup_event::wait(1, &event);
    }
  }
}

"""
  }
  
  func zeroInitialize(descriptor: AttentionHBMAccessDescriptor) -> String {
    guard let name = descriptor.name,
          let transposeState = descriptor.transposeState,
          let leadingDimension = descriptor.leadingDimension,
          let matrixDimension = descriptor.matrixDimension,
          let matrixOffset = descriptor.matrixOffset else {
      fatalError("Descriptor was incomplete.")
    }
    
    // 32 x 64 allocation in threadgroup memory
    // leading dimension = transposeState ? 32 : 64
    let leadingBlockDimension = transposeState ? UInt16(32) : UInt16(64)
    
    let loopBody = """

simdgroup_matrix_storage<float> \(name);
\(name) = simdgroup_matrix_storage<float>(0);

ushort2 origin(d, 0);
\(name).store(
  \(name)_block, \(leadingBlockDimension), origin, \(transposeState));

"""
    
    return """

{
  // Find where the \(name) data will be written to.
  ushort2 \(name)_block_offset(morton_offset.x, morton_offset.y + sidx * 8);
  auto \(name)_block = (threadgroup float*)(threadgroup_block);
  \(name)_block = simdgroup_matrix_storage<float>::apply_offset(
    \(name)_block, \(leadingBlockDimension),
    \(name)_block_offset, \(transposeState));
  
  // Outer loop over D.
#pragma clang loop unroll(full)
  for (ushort d_outer = 0; d_outer < D; d_outer += 64) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Iterate over the head dimension.
    if (\(paddedD) - d_outer >= 64) {
#pragma clang loop unroll(full)
      for (ushort d = 0; d < 64; d += 8) {
        \(loopBody)
      }
    } else {
#pragma clang loop unroll(full)
      for (ushort d = 0; d < \(paddedD) % 64; d += 8) {
        \(loopBody)
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    if (sidx == 0) {
      uint2 \(name)_offset(d_outer, \(matrixOffset));
      auto src = (threadgroup float*)(threadgroup_block);
      auto dst = simdgroup_matrix_storage<float>::apply_offset(
        \(name), \(leadingDimension), \(name)_offset, \(transposeState));
     
      ushort D_dimension = min(ushort(64), ushort(D - d_outer));
      ushort RC_dimension = min(
        uint(32), \(matrixDimension) - \(matrixOffset));
      ushort2 tile_src(D_dimension, RC_dimension);
      ushort2 tile_dst(D_dimension, RC_dimension);
      
      simdgroup_event event;
      event.async_copy(
        dst, \(leadingDimension), tile_dst,
        src, \(leadingBlockDimension), tile_src, \(transposeState));
      simdgroup_event::wait(1, &event);
    }
  }
}

"""
  }
}

// MARK: - Arguments

extension AttentionKernel {
  func createArguments(type: AttentionKernelType) -> String {
    struct AttentionOperand {
      var precision: GEMMOperandPrecision
      var bufferBinding: Int
    }
    
    // Index the operands available during the forward pass.
    var operandsMap: [String: AttentionOperand] = [:]
    operandsMap["Q"] = AttentionOperand(
      precision: .FP32, bufferBinding: 0)
    operandsMap["K"] = AttentionOperand(
      precision: .FP32, bufferBinding: 1)
    operandsMap["V"] = AttentionOperand(
      precision: .FP32, bufferBinding: 2)
    operandsMap["O"] = AttentionOperand(
      precision: .FP32, bufferBinding: 3)
    operandsMap["L_terms"] = AttentionOperand(
      precision: .FP32, bufferBinding: 4)
    
    // Index the operands available during the backward pass.
    operandsMap["dO"] = AttentionOperand(
      precision: .FP32, bufferBinding: 5)
    operandsMap["D_terms"] = AttentionOperand(
      precision: .FP32, bufferBinding: 6)
    operandsMap["dV"] = AttentionOperand(
      precision: .FP32, bufferBinding: 7)
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
      precision: .BF16, bufferBinding: 8)
    operandsMap["dK"] = AttentionOperand(
      precision: .FP32, bufferBinding: 8)
    operandsMap["dQ"] = AttentionOperand(
      precision: .FP32, bufferBinding: 9)
    
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

  uint linear_array_slot = gid * 32 + sidx * 8 + morton_offset.y;

"""
    default:
      break
    }
    
    return output
  }
}

extension AttentionKernel {
  func createSetup(type: AttentionKernelType) -> String {
    // Code that prepares an accumulator variable.
    func zeroInitializeAccumulator(name: String) -> String {
      return """

  simdgroup_matrix_storage<float> \(name)_sram[\(paddedD / 8)];
#pragma clang loop unroll(full)
  for (ushort d = 0; d < \(paddedD); d += 8) {
    \(name)_sram[d / 8] = simdgroup_matrix_storage<float>(0);
  }

"""
    }
    
    // Initialize the output string.
    var output: String = ""
    
    switch type {
    case .forward:
      if cachedInputs.Q {
        var accessDesc = AttentionHBMAccessDescriptor()
        accessDesc.name = "Q"
        accessDesc.transposeState = transposeState.Q
        accessDesc.leadingDimension = leadingDimensions.Q
        accessDesc.matrixDimension = "R"
        accessDesc.matrixOffset = "gid * 32"
        output += load(descriptor: accessDesc)
      }
      
      if cachedOutputs.O {
        output += zeroInitializeAccumulator(name: "O")
      } else {
        var accessDesc = AttentionHBMAccessDescriptor()
        accessDesc.name = "O"
        accessDesc.transposeState = transposeState.O
        accessDesc.leadingDimension = leadingDimensions.O
        accessDesc.matrixDimension = "R"
        accessDesc.matrixOffset = "gid * 32"
        output += zeroInitialize(descriptor: accessDesc)
      }
      
      output += """

  float m = -numeric_limits<float>::max();
  float l = numeric_limits<float>::denorm_min();

"""
      
    case .backwardQuery(let computeDerivativeQ):
      if cachedInputs.Q {
        var accessDesc = AttentionHBMAccessDescriptor()
        accessDesc.name = "Q"
        accessDesc.transposeState = transposeState.Q
        accessDesc.leadingDimension = leadingDimensions.Q
        accessDesc.matrixDimension = "R"
        accessDesc.matrixOffset = "gid * 32"
        output += load(descriptor: accessDesc)
      }
      
      if cachedInputs.dO {
        var accessDesc = AttentionHBMAccessDescriptor()
        accessDesc.name = "dO"
        accessDesc.transposeState = transposeState.O
        accessDesc.leadingDimension = leadingDimensions.O
        accessDesc.matrixDimension = "R"
        accessDesc.matrixOffset = "gid * 32"
        output += load(descriptor: accessDesc)
      }
      
      if computeDerivativeQ {
        output += zeroInitializeAccumulator(name: "dQ")
        output += """

  float L_term = L_terms[linear_array_slot];

"""
      }
      
      output += computeDTerm()
      
    case .backwardKeyValue(let computeDerivativeK):
      if cachedInputs.K {
        var accessDesc = AttentionHBMAccessDescriptor()
        accessDesc.name = "K"
        accessDesc.transposeState = transposeState.K
        accessDesc.leadingDimension = leadingDimensions.K
        accessDesc.matrixDimension = "C"
        accessDesc.matrixOffset = "gid * 32"
        output += load(descriptor: accessDesc)
      }
      
      if cachedInputs.V {
        var accessDesc = AttentionHBMAccessDescriptor()
        accessDesc.name = "V"
        accessDesc.transposeState = transposeState.V
        accessDesc.leadingDimension = leadingDimensions.V
        accessDesc.matrixDimension = "C"
        accessDesc.matrixOffset = "gid * 32"
        output += load(descriptor: accessDesc)
      }
      
      if computeDerivativeK {
        output += zeroInitializeAccumulator(name: "dK")
      }
      
      output += zeroInitializeAccumulator(name: "dV")
    }
    
    return output
  }
}

extension AttentionKernel {
  func createCleanup(type: AttentionKernelType) -> String {
    // A threadgroup barrier, formatted to match the correct indentation.
    func threadgroupBarrier() -> String {
      return """

    threadgroup_barrier(mem_flags::mem_threadgroup);

  """
    }
    
    // Initialize the output string.
    var output: String = ""
    
    switch type {
    case .forward(let computeL):
      // O
      if cachedOutputs.O {
        var accessDesc = AttentionHBMAccessDescriptor()
        accessDesc.name = "O"
        accessDesc.transposeState = transposeState.O
        accessDesc.leadingDimension = leadingDimensions.O
        accessDesc.matrixDimension = "R"
        accessDesc.matrixOffset = "gid * 32"
        output += store(descriptor: accessDesc)
      }
      
      // L[i]
      if computeL {
        output += """

    if (linear_array_slot < R) {
      \(computeLTerm())
      L_terms[linear_array_slot] = L_term;
    }

"""
      }
    
    case .backwardQuery(let computeDerivativeQ):
      // dQ
      if computeDerivativeQ {
        var accessDesc = AttentionHBMAccessDescriptor()
        accessDesc.name = "dQ"
        accessDesc.transposeState = transposeState.Q
        accessDesc.leadingDimension = leadingDimensions.Q
        accessDesc.matrixDimension = "R"
        accessDesc.matrixOffset = "gid * 32"
        output += store(descriptor: accessDesc)
      }
      
      // D[i]
      output += """

  if (linear_array_slot < R) {
    D_terms[linear_array_slot] = D_term;
  }

"""
      
    case .backwardKeyValue(let computeDerivativeK):
      // dK
      if computeDerivativeK {
        var accessDesc = AttentionHBMAccessDescriptor()
        accessDesc.name = "dK"
        accessDesc.transposeState = transposeState.K
        accessDesc.leadingDimension = leadingDimensions.K
        accessDesc.matrixDimension = "C"
        accessDesc.matrixOffset = "gid * 32"
        output += store(descriptor: accessDesc)
      }
      
      // dV
      do {
        var accessDesc = AttentionHBMAccessDescriptor()
        accessDesc.name = "dV"
        accessDesc.transposeState = transposeState.V
        accessDesc.leadingDimension = leadingDimensions.V
        accessDesc.matrixDimension = "C"
        accessDesc.matrixOffset = "gid * 32"
        output += store(descriptor: accessDesc)
      }
    }
    
    return output
  }
}
