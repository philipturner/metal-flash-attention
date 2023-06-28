//
//  Metal.swift
//  MetalFlashAttention
//
//  Created by Philip Turner on 6/27/23.
//

import AppleGPUInfo
import MetalPerformanceShadersGraph

struct MetalContext {
  static let global = MetalContext()
  
  var device: MTLDevice
  var infoDevice: GPUInfoDevice
  var graphDevice: MPSGraphDevice
  
  var library: MTLLibrary
  var commandQueue: MTLCommandQueue
  
  init() {
    self.device = MTLCopyAllDevices().first!
    self.infoDevice = try! GPUInfoDevice()
    self.graphDevice = MPSGraphDevice(mtlDevice: device)
    
    var libraryURL = Bundle.main.resourceURL!
    libraryURL.append(component: "lib")
    libraryURL.append(component: "libMetalFlashAttention.metallib")
    self.library = try! device.makeLibrary(URL: libraryURL)
    self.commandQueue = device.makeCommandQueue()!
  }
}

protocol MetalBackend: _TensorBackend {
  associatedtype Resource: AsyncResource
}

protocol AsyncResource {
  associatedtype Resource
  
  // A background thread calls this to write the finished resoure.
  func finish(resource: Resource)
  
  // Lazily blocks until the background thread finishes.
  var resource: Resource { get }
}

protocol MetalOperation {
  associatedtype Backend: MetalBackend
  associatedtype Encoder
  associatedtype Tensors
  
  // Make an async resource if the cache doesn't already contain it.
  func makeAsyncResource() -> Backend.Resource
  
  // Never called during ghost execution.
  func encode(encoder: Encoder, tensors: Tensors, resource: Backend.Resource)
}

struct OperationCache<Backend: MetalBackend> {
  var gemm: [GEMM_Parameters: Backend.Resource]
  
  // TODO: Throw an error when you forgot the ghost pass.
}
