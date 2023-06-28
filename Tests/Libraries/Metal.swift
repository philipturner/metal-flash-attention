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
  associatedtype _AsyncResource
}

protocol AsyncResource {
  associatedtype Resource
  
  // A background thread calls this to write the finished resoure.
  func finish(resource: Resource)
  
  // Lazily blocks until the background thread finishes.
  var resource: Resource { get }
}

struct OperationCache<Backend: MetalBackend> {
  var gemm: [GEMM_Parameters: Backend._AsyncResource]
  
  // TODO: Function that throws an error because you didn't run a ghost
// execution pass beforehand.
}
