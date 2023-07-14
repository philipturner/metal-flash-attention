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
  let expectedBackend: TensorBackend = .mps
  let actualBackend: TensorBackend = .mfa
  typealias Real = Float16
  
//  let R = 30
//  let C = 30
//  let H = 1
//  let D = 20
  let R = 33
  let C = 15
  let H = 3
  let D = 27
  let expected_Q = Tensor<Real>(
    shape: [R, H, D], randomUniform: 0..<1, backend: expectedBackend)
  let expected_K = Tensor<Real>(
    shape: [C, H, D], randomUniform: 0..<1, backend: expectedBackend)
  let expected_V = Tensor<Real>(
    shape: [C, H, D], randomUniform: 0..<1, backend: expectedBackend)
  #if false
  var expected_O = Tensor<Real>(
    zerosLike: [R, H, D], backend: expectedBackend)
  #else
  var expected_O = Tensor<Real>(
    zerosLike: [H, D, R], backend: expectedBackend)
  #endif
  
  let actual_Q = Tensor(copying: expected_Q, backend: actualBackend)
  let actual_K = Tensor(copying: expected_K, backend: actualBackend)
  let actual_V = Tensor(copying: expected_V, backend: actualBackend)
  var actual_O = Tensor(copying: expected_O, backend: actualBackend)
  
  _ExecutionContext.withDefaultBackend(expectedBackend) {
    _ExecutionContext.profileCommands {
      expected_O.attention(
        queries: expected_Q, keys: expected_K, values: expected_V,
        transposeK: true, transposeO: true)
    }
  }
  
  _ExecutionContext.withDefaultBackend(actualBackend) {
    _ExecutionContext.profileCommands {
      actual_O.attention(
        queries: actual_Q, keys: actual_K, values: actual_V,
        transposeK: true, transposeO: true)
    }
  }
  
  for plane in 0..<H {
    #if false
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
      actual: actual_O, expected: expected_O,
      parameters: parameters, slice: plane, transpose: true)
    #endif
  }
}
