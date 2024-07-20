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
