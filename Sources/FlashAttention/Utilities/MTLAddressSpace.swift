//
//  MTLAddressSpace.swift
//  FlashAttention
//
//  Created by Philip Turner on 8/9/24.
//

public enum MTLAddressSpace {
  case device
  case threadgroup
  
  public var keyword: String {
    switch self {
    case .device: return "device"
    case .threadgroup: return "threadgroup"
    }
  }
  
  public var offsetType: String {
    switch self {
    case .device: return "uint"
    case .threadgroup: return "ushort"
    }
  }
}
