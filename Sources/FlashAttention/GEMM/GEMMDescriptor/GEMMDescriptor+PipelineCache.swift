//
//  PipelineCache.swift
//  FlashAttention
//
//  Created by Philip Turner on 6/21/24.
//

import Metal
import MetalASM

extension GEMMKernel {
  public typealias PipelineValue = (
    kernel: GEMMKernel, pipeline: MTLComputePipelineState)

  public static var pipelineCache: [
    GEMMDescriptor: PipelineValue] = [:]
}

extension GEMMKernel {
  // Register this problem configuration in the cache.
  public static func register(descriptor: GEMMDescriptor) {
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

    /// Create a monolithic pipeline via MetalASM.
    func createMonolithicPipeline(
      _ kernelDescriptor: GEMMKernelDescriptor
    ) -> PipelineValue {
      let kernel = GEMMKernel(descriptor: kernelDescriptor)

      // Build the MonolithicDescriptor from the GEMMDescriptor.
      guard let matrixDimensions = descriptor.matrixDimensions,
            let transposeState = descriptor.transposeState else {
        fatalError("Descriptor was incomplete.")
      }

      var monoDesc = GEMMKernel.MonolithicDescriptor()
      monoDesc.M = matrixDimensions.M
      monoDesc.N = matrixDimensions.N
      monoDesc.K = matrixDimensions.K
      monoDesc.loadPreviousC = descriptor.loadPreviousC

      // Compute leading dimensions (same logic as setFunctionConstants).
      func chooseLeadingDimension(
        _ specifiedLeading: UInt32?,
        _ transposeState: Bool,
        _ untransposedRows: UInt32,
        _ untransposedColumns: UInt32
      ) -> UInt32 {
        var expectedLeading: UInt32
        if transposeState {
          expectedLeading = untransposedRows
        } else {
          expectedLeading = untransposedColumns
        }
        if let specifiedLeading {
          guard specifiedLeading >= expectedLeading else {
            fatalError("Leading dimension was too small.")
          }
          return specifiedLeading
        }
        return expectedLeading
      }
      monoDesc.leadingDimensionA = chooseLeadingDimension(
        descriptor.leadingDimensions?.A, transposeState.A,
        matrixDimensions.M, matrixDimensions.K)
      monoDesc.leadingDimensionB = chooseLeadingDimension(
        descriptor.leadingDimensions?.B, transposeState.B,
        matrixDimensions.K, matrixDimensions.N)
      monoDesc.leadingDimensionC = chooseLeadingDimension(
        descriptor.leadingDimensions?.C, false,
        matrixDimensions.M, matrixDimensions.N)

      // Generate monolithic LLVM IR and assemble in-process.
      let ir = kernel.createSource(descriptor: monoDesc)


      #if os(macOS)
      let metallibData = try! MetalASM.assemble(ir: ir, platform: .macOS(version: 26))
      #elseif os(iOS)
      let metallibData = try! MetalASM.assemble(ir: ir, platform: .iOS(version: 26))
      #endif

      let dispatchData = metallibData.withUnsafeBytes {
        DispatchData(bytes: $0)
      }
      let library = try! device.makeLibrary(data: dispatchData)
      let function = library.makeFunction(name: "gemm")!
      let pipeline = try! device.makeComputePipelineState(function: function)

      return (kernel, pipeline)
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

        let pipelineValue = createMonolithicPipeline(modifiedKernelDescriptor)
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
      let pipelineValue = createMonolithicPipeline(kernelDescriptor)
      GEMMKernel.pipelineCache[descriptor] = pipelineValue
    }
  }
}
