//
//  ShaderCache.swift
//  FlashAttention
//
//  Created by Philip Turner on 6/21/24.
//

import Metal

/// A reference implementation of shader caching.
///
/// One good design for a shader caching mechanism:
/// - Two key-value caches.
/// - The first caches `MTLLibrary` objects.
///   - Large latency
///   - Small number of combinatorial possibilities, likely to be shared by
///     matrices with a different size.
///   - Don't bother with serializing Metal binary archives to disk. You are
///     already utilizing the system-wide Metal shader cache.
/// - The second caches `MTLComputePipelineState` objects.
///   - Instantiations of the `MTLLibrary` with different function constants.
///   - Less latency than compiling from source, but still non-negligible. You
///     can't spawn a new PSO during every call to a matrix multiplication.
extension GEMMKernel {
  /// WARNING: Not thread safe. But will the DSL interpreter even use
  /// multithreading?
  static var libraryCache: [
    GEMMKernelDescriptor: GEMMKernel] = [:]
  
  /// WARNING: Not thread safe. But will the DSL interpreter even use
  /// multithreading?
  static var pipelineCache: [
    GEMMDescriptor: (GEMMKernel, MTLComputePipelineState)] = [:]
}

extension GEMMKernel {
  /// Implementation of the logic for choosing between 'device' and
  /// 'threadgroup' store.
  static func fetchKernel(
    descriptor gemmDesc: GEMMDescriptor
  ) -> (GEMMKernel, MTLComputePipelineState) {
    // Perform the early return before anything with high latency.
    if let value = GEMMKernel.pipelineCache[gemmDesc] {
      return value
    }
    func createKernel(descriptor: GEMMKernelDescriptor) -> GEMMKernel {
      guard descriptor.preferAsyncStore != nil else {
        fatalError("Prefer async store was not set.")
      }
      if let previous = GEMMKernel.libraryCache[descriptor] {
        return previous
      } else {
        return GEMMKernel(descriptor: descriptor)
      }
    }
    
    // Create a MTLDevice object, a function call with very high latency.
    let device = MTLCreateSystemDefaultDevice()!
    func createPipeline(library: MTLLibrary) -> MTLComputePipelineState {
      // Set the function constants.
      let constants = MTLFunctionConstantValues()
      var M = gemmDesc.matrixDimensions!.M
      var N = gemmDesc.matrixDimensions!.N
      var K = gemmDesc.matrixDimensions!.K
      constants.setConstantValue(&M, type: .uint, index: 0)
      constants.setConstantValue(&N, type: .uint, index: 1)
      constants.setConstantValue(&K, type: .uint, index: 2)
      
      var loadPreviousC = gemmDesc.loadPreviousC
      constants.setConstantValue(&loadPreviousC, type: .bool, index: 10)
      
      let function = try! library.makeFunction(
        name: "gemm", constantValues: constants)
      let pipeline = try! device.makeComputePipelineState(function: function)
      return pipeline
    }
    
    var kernelDesc = GEMMKernelDescriptor(descriptor: gemmDesc)
    kernelDesc.device = device
    if device.supportsFamily(.apple9) {
      kernelDesc.preferAsyncStore = false
    } else {
      guard let blockDimensions = kernelDesc.blockDimensions else {
        fatalError("Block dimensions were not set.")
      }
      if blockDimensions == (48, 48, 32) {
        kernelDesc.preferAsyncStore = nil
      } else {
        kernelDesc.preferAsyncStore = true
      }
    }
    
    var output: (GEMMKernel, MTLComputePipelineState)
    if kernelDesc.preferAsyncStore != nil {
      let kernel = createKernel(descriptor: kernelDesc)
      let pipeline = createPipeline(library: kernel.library)
      output = (kernel, pipeline)
      
      GEMMKernel.libraryCache[kernelDesc] = kernel
    } else {
      
      var candidates: [
        (kernelDesc: GEMMKernelDescriptor,
         kernel: GEMMKernel,
         pipeline: MTLComputePipelineState)
      ] = []
      for candidateID in 0..<4 {
        var blockDimensions: (M: UInt16, N: UInt16, K: UInt16)
        var preferAsyncStore: Bool
        switch candidateID {
        case 0:
          blockDimensions = (48, 48, 32)
          preferAsyncStore = false
        case 1:
          blockDimensions = (48, 48, 40)
          preferAsyncStore = false
        case 2:
          blockDimensions = (48, 48, 32)
          preferAsyncStore = true
        case 3:
          blockDimensions = (48, 48, 40)
          preferAsyncStore = true
        default:
          fatalError("This should never happen.")
        }
        
        // Set the data that's unique to this variant.
        var newKernelDesc = kernelDesc
        newKernelDesc.blockDimensions = blockDimensions
        newKernelDesc.preferAsyncStore = preferAsyncStore
        
        let kernel = createKernel(descriptor: newKernelDesc)
        let pipeline = createPipeline(library: kernel.library)
        candidates.append((newKernelDesc, kernel, pipeline))
        
        GEMMKernel.libraryCache[newKernelDesc] = kernel
      }
      
      // Find the maximum occupancy.
      var maximumOccupancy: Int = -1
      for candidate in candidates {
        let occupancy = candidate.pipeline.maxTotalThreadsPerThreadgroup
        maximumOccupancy = max(maximumOccupancy, occupancy)
      }
      candidates.removeAll(where: {
        $0.pipeline.maxTotalThreadsPerThreadgroup != maximumOccupancy
      })
      
      // Choose the highest-performing candidate.
      let candidate = candidates.last!
      kernelDesc = candidate.kernelDesc
      output = (candidate.kernel, candidate.pipeline)
    }
    
    // Save the output to the cache.
    GEMMKernel.pipelineCache[gemmDesc] = output
    return output
  }
}
