//
//  ShaderCache.swift
//  FlashAttention
//
//  Created by Philip Turner on 6/21/24.
//

import Metal

extension GEMMKernel {
  typealias LibraryValue = (
    kernel: GEMMKernel, library: MTLLibrary)
  typealias PipelineValue = (
    kernel: GEMMKernel, pipeline: MTLComputePipelineState)
  
  static var libraryCache: [
    GEMMKernelDescriptor: LibraryValue] = [:]
  static var pipelineCache: [
    GEMMDescriptor: PipelineValue] = [:]
}

extension GEMMKernel {
  // Register this problem configuration in the cache.
  static func register(descriptor: GEMMDescriptor) {
    guard pipelineCache[descriptor] == nil else {
      return
    }
    
    var kernelDescriptor = GEMMKernelDescriptor(descriptor: descriptor)
    
    let device = MTLContext.global.device
    if device.supportsFamily(.apple9) {
      kernelDescriptor.preferAsyncStore = false
    } else {
      guard let blockDimensions = kernelDescriptor.blockDimensions else {
        fatalError("Block dimensions were not set.")
      }
      if blockDimensions == (48, 48, 32) {
        kernelDescriptor.preferAsyncStore = nil
      } else {
        kernelDescriptor.preferAsyncStore = true
      }
    }
    
    func createLibrary(
      _ kernelDescriptor: GEMMKernelDescriptor
    ) -> LibraryValue {
      if let output = GEMMKernel.libraryCache[kernelDescriptor] {
        return output
      } else {
        let kernel = GEMMKernel(descriptor: kernelDescriptor)
        let source = kernel.createSource()
        let library = try! device.makeLibrary(source: source, options: nil)
        
        let output = (kernel, library)
        GEMMKernel.libraryCache[kernelDescriptor] = output
        return output
      }
    }
    
    func createPipeline(
      _ libraryValue: LibraryValue
    ) -> PipelineValue {
      let constants = MTLFunctionConstantValues()
      descriptor.setFunctionConstants(constants)
      
      let library = libraryValue.library
      let function = try! library.makeFunction(
        name: "gemm", constantValues: constants)
      let pipeline = try! device.makeComputePipelineState(
        function: function)
      return (libraryValue.kernel, pipeline)
    }
    
    if kernelDescriptor.preferAsyncStore == nil {
      var candidates: [PipelineValue] = []
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
        
        // Set the attributes unique to this variant.
        var modifiedKernelDescriptor = kernelDescriptor
        modifiedKernelDescriptor.blockDimensions = blockDimensions
        modifiedKernelDescriptor.preferAsyncStore = preferAsyncStore
        
        let libraryValue = createLibrary(modifiedKernelDescriptor)
        let pipelineValue = createPipeline(libraryValue)
        candidates.append(pipelineValue)
      }
      
      // Find the maximum occupancy.
      var maximumOccupancy: Int = -1
      for candidate in candidates {
        let pipeline = candidate.pipeline
        let occupancy = pipeline.maxTotalThreadsPerThreadgroup
        maximumOccupancy = max(maximumOccupancy, occupancy)
      }
      candidates.removeAll(where: {
        $0.pipeline.maxTotalThreadsPerThreadgroup != maximumOccupancy
      })
      
      // Choose the highest-performing candidate.
      GEMMKernel.pipelineCache[descriptor] = candidates.last!
    } else {
      let libraryValue = createLibrary(kernelDescriptor)
      let pipelineValue = createPipeline(libraryValue)
      GEMMKernel.pipelineCache[descriptor] = pipelineValue
    }
  }
}
