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
  /// Name of the output, destination of a 32 x D block.
  var O: String?
  
  var transposeO: Bool?
  var leadingDimensionO: String?
  
  var matrixDimension: String?
  var matrixOffset: String?
}

extension AttentionKernel {
  func store(descriptor: AttentionHBMAccessDescriptor) -> String {
    guard let O = descriptor.O,
          let transposeO = descriptor.transposeO,
          let leadingDimensionO = descriptor.leadingDimensionO,
          let matrixDimension = descriptor.matrixDimension,
          let matrixOffset = descriptor.matrixOffset else {
      fatalError("Descriptor was incomplete.")
    }
    
    // 32 x 64 allocation in threadgroup memory
    // leading dimension = transposeO ? 32 : 64
    let leadingBlockDimensionO = transposeO ? UInt16(32) : UInt16(64)
    
    let loopBodyStoreO = """

ushort2 origin(d, 0);
\(O)_sram[(d_outer + d) / 8].store(
  \(O)_block, \(leadingBlockDimensionO), origin, \(transposeO));

"""
    
    return """

{
  // Find where the \(O) data will be written to.
  ushort2 \(O)_block_offset(morton_offset.x, morton_offset.y + sidx * 8);
  auto \(O)_block = (threadgroup float*)(threadgroup_block);
  \(O)_block = simdgroup_matrix_storage<float>::apply_offset(
    \(O)_block, \(leadingBlockDimensionO),
    \(O)_block_offset, \(transposeO));
  
  // Outer loop over D.
#pragma clang loop unroll(full)
  for (ushort d = 0; d < D; d += 64) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Iterate over the head dimension.
    ushort d_outer = d;
    if (\(paddedD) - d_outer >= 64) {
#pragma clang loop unroll(full)
      for (ushort d = 0; d < 64; d += 8) {
        \(loopBodyStoreO)
      }
    } else {
#pragma clang loop unroll(full)
      for (ushort d = 0; d < \(paddedD) % 64; d += 8) {
        \(loopBodyStoreO)
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    if (sidx == 0) {
      uint2 \(O)_offset(d, \(matrixOffset));
      auto src = (threadgroup float*)(threadgroup_block);
      auto dst = simdgroup_matrix_storage<float>::apply_offset(
        \(O), \(leadingDimensionO), \(O)_offset, \(transposeO));
     
      ushort D_dimension = min(ushort(64), ushort(D - d));
      ushort RC_dimension = min(
        uint(32), \(matrixDimension) - \(matrixOffset));
      ushort2 tile_src(D_dimension, RC_dimension);
      ushort2 tile_dst(D_dimension, RC_dimension);
      
      simdgroup_event event;
      event.async_copy(
        dst, \(leadingDimensionO), tile_dst,
        src, \(leadingBlockDimensionO), tile_src, \(transposeO));
      simdgroup_event::wait(1, &event);
    }
  }
}

"""
  }
}

// MARK: - Setup and Cleanup

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
      output += zeroInitializeAccumulator(name: "O")
      output += """

  float m = -numeric_limits<float>::max();
  float l = numeric_limits<float>::denorm_min();

"""
      
    case .backwardQuery(let computeDerivativeQ):
      if computeDerivativeQ {
        output += zeroInitializeAccumulator(name: "dQ")
        output += """

  float L_term = L_terms[linear_array_slot];

"""
      }
      output += computeDTerm()
      
    case .backwardKeyValue(let computeDerivativeK):
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
      var accessDesc = AttentionHBMAccessDescriptor()
      accessDesc.O = "O"
      accessDesc.transposeO = transposeState.O
      accessDesc.leadingDimensionO = leadingDimensions.O
      accessDesc.matrixDimension = "R"
      accessDesc.matrixOffset = "gid * R_group"
      
      output += store(descriptor: accessDesc)
      
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
        accessDesc.O = "dQ"
        accessDesc.transposeO = transposeState.Q
        accessDesc.leadingDimensionO = leadingDimensions.Q
        accessDesc.matrixDimension = "R"
        accessDesc.matrixOffset = "gid * R_group"
        
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
        accessDesc.O = "dK"
        accessDesc.transposeO = transposeState.K
        accessDesc.leadingDimensionO = leadingDimensions.K
        accessDesc.matrixDimension = "C"
        accessDesc.matrixOffset = "gid * C_group"
        
        output += store(descriptor: accessDesc)
      }
      
      // dV
      do {
        var accessDesc = AttentionHBMAccessDescriptor()
        accessDesc.O = "dV"
        accessDesc.transposeO = transposeState.V
        accessDesc.leadingDimensionO = leadingDimensions.V
        accessDesc.matrixDimension = "C"
        accessDesc.matrixOffset = "gid * C_group"
        
        output += store(descriptor: accessDesc)
      }
    }
    
    return output
  }
}
