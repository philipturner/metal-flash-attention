//
//  AttentionKernel+Load.swift
//  FlashAttention
//
//  Created by Philip Turner on 7/19/24.
//

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
    
    // A threadgroup barrier, formatted to match the correct indentation.
    func threadgroupBarrier() -> String {
      return """

    threadgroup_barrier(mem_flags::mem_threadgroup);

  """
    }
    
    // Initialize the output string.
    var output: String = ""
    
    // Loading everything that could possibly be loaded, for now.
    switch type {
    case .forward:
      // O, m, l
      output += zeroInitializeAccumulator(name: "O")
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
        
        output += computeDTerm()
      }
      
      // dQ, L[i]
      if computeDerivativeQ {
        output += zeroInitializeAccumulator(name: "dQ")
        output += """

  float L_term = L_terms[linear_array_slot];

"""
      }
      
    case .backwardKeyValue(let computeDerivativeK):
      var accessDesc = AttentionHBMAccessDescriptor()
      accessDesc.index = "gid * C_group"
      accessDesc.leadingBlockDimension = leadingBlockDimensions.V
      accessDesc.leadingDimension = leadingDimensions.V
      accessDesc.name = "V"
      accessDesc.threadgroupAddress = "threadgroup_block"
      accessDesc.transposeState = transposeState.V
      
      // dK, dV, V
      if computeDerivativeK {
        output += zeroInitializeAccumulator(name: "dK")
      }
      output += prefetchColumns(descriptor: accessDesc)
      output += zeroInitializeAccumulator(name: "dV")
      output += threadgroupBarrier()
      output += load(descriptor: accessDesc)
    }
    
    return output
  }
}

// TODO: Delete the functions below, when the remaining code switches to the
// blocked algorithm.

extension AttentionKernel {
  func blockQ() -> String {
    "(threadgroup float*)(threadgroup_block)"
  }
  
  func blockLTerms() -> String {
    if transposeState.Q {
      // D x R, where R is the row stride.
      return "\(blockQ()) + \(paddedD) * \(leadingBlockDimensions.Q)"
    } else {
      // R x D, where D is the row stride.
      return "\(blockQ()) + R_group * \(leadingBlockDimensions.Q)"
    }
  }
  
  func prefetchQLTerms() -> String {
    return """
    
    if (sidx == 0) {
      uint2 device_origin(0, r);
      auto Q_src = simdgroup_matrix_storage<float>::apply_offset(
        Q, \(leadingDimensions.Q), device_origin, \(transposeState.Q));
      auto Q_dst = \(blockQ());
      auto L_terms_src = L_terms + r;
      auto L_terms_dst = \(blockLTerms());
      
      // Zero-padding for safety, which should harm performance.
      ushort R_tile_dimension = min(uint(R_group), R - r);
      ushort2 tile_src(D, R_tile_dimension);
      ushort2 tile_dst(\(paddedD), R_group);
      
      // Issue two async copies.
      simdgroup_event events[2];
      events[0].async_copy(
        Q_dst, \(leadingBlockDimensions.Q), tile_dst,
        Q_src, \(leadingDimensions.Q), tile_src, \(transposeState.Q));
      events[1].async_copy(
        L_terms_dst, 1, ushort2(tile_dst.y, 1),
        L_terms_src, 1, ushort2(tile_src.y, 1));
      simdgroup_event::wait(2, events);
    }

"""
  }
  
  func blockDerivativeO() -> String {
    "(threadgroup float*)(threadgroup_block)"
  }
  
  func blockDTerms() -> String {
    if transposeState.O {
      // D x R, where R is the row stride.
      return "\(blockDerivativeO()) + \(paddedD) * \(leadingBlockDimensions.O)"
    } else {
      // R x D, where D is the row stride.
      return "\(blockDerivativeO()) + R_group * \(leadingBlockDimensions.O)"
    }
  }
  
  func prefetchDerivativeODTerms() -> String {
    return """
    
    if (sidx == 0) {
      uint2 device_origin(0, r);
      auto dO_src = simdgroup_matrix_storage<float>::apply_offset(
        dO, \(leadingDimensions.O), device_origin, \(transposeState.O));
      auto dO_dst = \(blockDerivativeO());
      auto D_terms_src = D_terms + r;
      auto D_terms_dst = \(blockDTerms());
      
      // Zero-padding for safety, which should harm performance.
      ushort R_tile_dimension = min(uint(R_group), R - r);
      ushort2 tile_src(D, R_tile_dimension);
      ushort2 tile_dst(\(paddedD), R_group);
      
      // Issue two async copies.
      simdgroup_event events[2];
      events[0].async_copy(
        dO_dst, \(leadingBlockDimensions.O), tile_dst,
        dO_src, \(leadingDimensions.O), tile_src, \(transposeState.O));
      events[1].async_copy(
        D_terms_dst, 1, ushort2(tile_dst.y, 1),
        D_terms_src, 1, ushort2(tile_src.y, 1));
      simdgroup_event::wait(2, events);
    }

"""
  }
}
