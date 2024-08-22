//
//  MTLContext.swift
//  FlashAttention
//
//  Created by Philip Turner on 6/26/24.
//

import Metal

public struct MTLContext {
  public var device: MTLDevice
  public var commandQueue: MTLCommandQueue
  
  public static let global = MTLContext()
  
  public init() {
    device = MTLCreateSystemDefaultDevice()!
    commandQueue = device.makeCommandQueue()!
  }
}
