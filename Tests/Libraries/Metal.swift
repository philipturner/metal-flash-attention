//
//  Metal.swift
//  MetalFlashAttention
//
//  Created by Philip Turner on 6/27/23.
//

import Metal

struct MetalContext {
  static let global = MetalContext()
  
  var device: MTLDevice
  var commandQueue: MTLCommandQueue
  var library: MTLLibrary
  
  init() {
    self.device = MTLCopyAllDevices().first!
    self.commandQueue = device.makeCommandQueue()!
    
    var libraryURL = Bundle.main.resourceURL!
    libraryURL.append(component: "lib")
    libraryURL.append(component: "libMetalFlashAttention.metallib")
    self.library = try! device.makeLibrary(URL: libraryURL)
  }
}
