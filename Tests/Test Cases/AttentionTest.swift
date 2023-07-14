//
//  MaskTest.swift
//  MetalFlashAttention
//
//  Created by Philip Turner on 7/14/23.
//

import Foundation
import PythonKit

func showMaskTest() {
  let maskIsTriangular = Float.random(in: 0..<1) < 2 // suppress warning
  let maskPlane = 2
  typealias Real = Float16
  srand48(79)
  
  var mask: AttentionMask
  var R: Int
  var C: Int
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

func showAttentionTest() {
  let expectedBackend: TensorBackend = .numpy
  let actualBackend: TensorBackend = .mps
  typealias Real = Float32
  
  // After getting this to work, try H=2, which will have 2 planes.
  let R = 5
  let C = 4
  let H = 1
  let D = 7
  let expected_Q = Tensor<Real>(
    shape: [R, H, D], randomUniform: 0..<1, backend: expectedBackend)
  let expected_K = Tensor<Real>(
    shape: [C, H, D], randomUniform: 0..<1, backend: expectedBackend)
  let expected_V = Tensor<Real>(
    shape: [C, H, D], randomUniform: 0..<1, backend: expectedBackend)
  var expected_O = Tensor<Real>(
    zerosLike: [H, D, R], backend: expectedBackend)
  
  let actual_Q = Tensor(copying: expected_Q, backend: actualBackend)
  let actual_K = Tensor(copying: expected_K, backend: actualBackend)
  let actual_V = Tensor(copying: expected_V, backend: actualBackend)
  var actual_O = Tensor(copying: expected_O, backend: actualBackend)
  
  _ExecutionContext.withDefaultBackend(expectedBackend) {
    _ExecutionContext.profileCommands {
      expected_O.attention(
        queries: expected_Q, keys: expected_K, values: expected_V,
        transposeO: true)
    }
  }
  
  _ExecutionContext.withDefaultBackend(actualBackend) {
    _ExecutionContext.profileCommands {
      actual_O.attention(
        queries: actual_Q, keys: actual_K, values: actual_V,
        transposeO: true)
    }
  }
  
  for plane in 0..<H {
    let parameters = EuclideanDistanceParameters(
      attentionC: C, attentionH: H, attentionD: D)
    let plane = PythonObject(plane)
    MPL_showComparison(
      actual: actual_O, expected: expected_O, parameters: parameters,
      slice: plane, transpose: true)
  }
}