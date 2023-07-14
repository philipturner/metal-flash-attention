//
//  MaskTest.swift
//  MetalFlashAttention
//
//  Created by Philip Turner on 7/14/23.
//

import Foundation
import PythonKit

func showMaskTest() {
  let maskIsTriangular = false
  let maskPlane = 2
  typealias Real = Float16
  srand48(79)
  
  var mask: AttentionMask
  var R: Int
  var C: Int
  // Actual masks are 4D with H=1, making visualization impossible.
  let B = 3
  if maskIsTriangular {
    mask = .upperTriangular
    R = 13
    C = 13
  } else {
    mask = .blockSparse(3, 0.2)
    R = 16
    C = 20
  }
  
  let shape: [Int] = [B, R, C]
  let py_maskTensor = Tensor<Real>(shape: shape, mask: mask, backend: .numpy)
  let mps_maskTensor = Tensor(copying: py_maskTensor, backend: .mps)
  let mfa_maskTensor = Tensor(copying: py_maskTensor, backend: .mfa)
  
  let parameters = EuclideanDistanceParameters(
    averageMagnitude: 1.0, averageDeviation: 1.0, batchSize: B, bias: -0.5)
  let slice = PythonObject(maskPlane)
  MPL_showBackends(
    mfa: mfa_maskTensor,
    mps: mps_maskTensor,
    numpy: py_maskTensor,
    parameters: parameters,
    slice: slice)
}
