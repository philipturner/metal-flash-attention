//
//  MTLAddressSpace.swift
//  FlashAttention
//
//  Created by Philip Turner on 8/9/24.
//

enum MTLAddressSpace {
  case device
  case threadgroup
  
  var keyword: String {
    switch self {
    case .device: return "device"
    case .threadgroup: return "threadgroup"
    }
  }
  
  var offsetType: String {
    switch self {
    case .device: return "uint"
    case .threadgroup: return "ushort"
    }
  }
}
