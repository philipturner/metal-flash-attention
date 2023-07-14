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
  let actualBackend: TensorBackend = .mfa
  typealias Real = Float32
  
//  let R = 30
//  let C = 30
//  let H = 1
//  let D = 20
  let R = 27
  let C = 27
  let H = 1
  let D = 27
  let expected_Q = Tensor<Real>(
    shape: [R, H, D], randomUniform: 0..<1, backend: expectedBackend)
  let expected_K = Tensor<Real>(
    shape: [H, D, C], randomUniform: 0..<1, backend: expectedBackend)
  let expected_V = Tensor<Real>(
    shape: [C, H, D], randomUniform: 0..<1, backend: expectedBackend)
  var expected_O = Tensor<Real>(
    zerosLike: [R, H, D], backend: expectedBackend)
  
  let actual_Q = Tensor(copying: expected_Q, backend: actualBackend)
  let actual_K = Tensor(copying: expected_K, backend: actualBackend)
  let actual_V = Tensor(copying: expected_V, backend: actualBackend)
  var actual_O = Tensor(copying: expected_O, backend: actualBackend)
  
  _ExecutionContext.withDefaultBackend(expectedBackend) {
    _ExecutionContext.profileCommands {
      expected_O.attention(
        queries: expected_Q, keys: expected_K, values: expected_V,
        transposeK: false, transposeO: false)
    }
  }
  
  _ExecutionContext.withDefaultBackend(actualBackend) {
    _ExecutionContext.profileCommands {
      actual_O.attention(
        queries: actual_Q, keys: actual_K, values: actual_V,
        transposeK: false, transposeO: false)
    }
  }
  
  for plane in 0..<H {
    #if true
    let parameters = EuclideanDistanceParameters(
      averageMagnitude: 1,//Float(D) * Float(C) * 1, // 1
      averageDeviation: 0.2,//sqrt(Float(D)), // 0.2
      batchSize: nil)
    let plane = PythonObject(plane)
    
    let actual_reshaped = Tensor(
      shape: [R, D], reshaping: actual_O, backend: actualBackend)
    let expected_reshaped = Tensor(
      shape: [R, D], reshaping: expected_O, backend: expectedBackend)
    MPL_showComparison(
      actual: actual_reshaped, expected: expected_reshaped,
      parameters: parameters)
    #else
    let parameters = EuclideanDistanceParameters(
      attentionC: C, attentionH: H, attentionD: D)
    let plane = PythonObject(plane)
    
    MPL_showComparison(
      actual: actual_reshaped, expected: expected_reshaped,
      parameters: parameters, slice: plane, transpose: false)
    #endif
  }
}
