//
//  DataType.swift
//  MetalFlashAttention
//
//  Created by Philip Turner on 6/27/23.
//

import Metal
import PythonKit

extension MTLDataType {
  private func unrecognizedError() -> Never {
    fatalError("MTLDataType with code \(self.rawValue) not recognized.")
  }
  
  var numpy: PythonObject {
    let ctx = PythonContext.global
    switch self {
    case .half:
      return ctx.np.float16
    case .float:
      return ctx.np.float32
    default:
      unrecognizedError()
    }
  }
  
  var size: Int {
    switch self {
    case .half:
      return 2
    case .float:
      return 4
    default:
      unrecognizedError()
    }
  }
}
