//
//  Python.swift
//  MetalFlashAttention
//
//  Created by Philip Turner on 6/27/23.
//

import Foundation
import PythonKit

struct PythonContext {
  static let global = PythonContext()
  
  var ctypes: PythonObject
  var np: PythonObject
  var plt: PythonObject
  var mpl: PythonObject
  
  init() {
    let downloadsURL = FileManager.default.urls(
      for: .downloadsDirectory, in: .userDomainMask)[0]
    let homePath = downloadsURL.deletingLastPathComponent().relativePath
    let packages = homePath + "/miniforge3/lib/python3.11/site-packages"
    setenv("PYTHONPATH", packages, 1)
    
    self.ctypes = Python.import("ctypes")
    self.np = Python.import("numpy")
    self.plt = Python.import("matplotlib.pyplot")
    self.mpl = Python.import("matplotlib")
  }
}
