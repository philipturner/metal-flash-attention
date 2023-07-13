//
//  Attention.swift
//  MetalFlashAttention
//
//  Created by Philip Turner on 6/27/23.
//

import Metal
import MetalPerformanceShadersGraph
import PythonKit

protocol Attention: Operation {
  
}

// Broadcasting only supported along the mask.
struct Attention_Parameters: Hashable, Equatable {
  var dataType: MTLDataType
  var R: Int
  var C: Int
  var H: Int
  var D: Int
  var Q_trans: Bool
  var K_trans: Bool
  var V_trans: Bool
  var O_trans: Bool
  var batched: Bool
  var masked: Bool
  var block_sparse: Bool
  
  // These are only needed by MPSGraph; MFA supports dynamic batch size.
  var batchDimensionsQ: [Int]?
  var batchDimensionsMask: [Int]?
}

struct Attention_Tensors {
  var q: TensorBuffer
  var k: TensorBuffer
  var v: TensorBuffer
  var o: TensorBuffer
  var mask: TensorBuffer?
}

struct MFA_Attention {
  var parameters: Attention_Parameters
  
  static var functionConstants: [String: MTLConvertible] = [
    "R_simd": UInt16(16),
    "C_simd": UInt16(64),
    "R_splits": UInt16(4),
  ]
  
  init(parameters: Attention_Parameters) {
    self.parameters = parameters
  }
}

// TODO: Implement MPSGraph and NumPy einsum attention before MFA attention

struct MPS_Attention {
  var parameters: Attention_Parameters
  
  init(parameters: Attention_Parameters) {
    self.parameters = parameters
  }
  
  func makeAsyncResource() -> AsyncGraph {
    let dataType = parameters.dataType
    precondition(dataType == .float || dataType == .half)
    if parameters.batched {
      precondition(parameters.batchDimensionsQ!.count > 0)
      if parameters.masked {
        precondition(parameters.batchDimensionsMask!.count > 0)
        if let batchDimensionsMask = parameters.batchDimensionsMask {
          precondition(
            batchDimensionsMask.reduce(1, *) == 1 ||
            batchDimensionsMask == parameters.batchDimensionsQ!)
        }
      }
    } else {
      precondition(parameters.batchDimensionsQ! == [])
      if parameters.masked {
        precondition(parameters.batchDimensionsMask! == [])
      }
    }
    if !parameters.masked {
      precondition(parameters.batchDimensionsMask == nil)
    }
    
    let qBatch: [Int] = parameters.batchDimensionsQ!
    let maskBatch: [Int]? = parameters.batchDimensionsMask
    
    let qShape: [Int] = qBatch + [parameters.R, parameters.H, parameters.D]
    let kShape: [Int] = qBatch + [parameters.H, parameters.D, parameters.C]
    let vShape: [Int] = qBatch + [parameters.C, parameters.H, parameters.D]
    let oShape: [Int] = qBatch + [parameters.R, parameters.H, parameters.D]
    var maskShape: [Int]?
    if let maskBatch {
      maskShape = maskBatch + [1, parameters.R, parameters.C]
    }
    
    var qShapeTranspose: [Int]?
    var kShapeTranspose: [Int]?
    var vShapeTranspose: [Int]?
    var oShapeTranspose: [Int]?
    if parameters.Q_trans {
      qShapeTranspose = qBatch + [parameters.H, parameters.D, parameters.R]
    }
    if parameters.K_trans {
      kShapeTranspose = qBatch + [parameters.C, parameters.H, parameters.D]
    }
    if parameters.V_trans {
      vShapeTranspose = qBatch + [parameters.H, parameters.D, parameters.C]
    }
    if parameters.O_trans {
      oShapeTranspose = qBatch + [parameters.H, parameters.D, parameters.R]
    }
    let graph = MPSGraph()
    func shape(_ shape: [Int]?) -> [Int] {
      shape!
    }
    func nsShape(_ shape: [Int]?) -> [NSNumber] {
      shape!.map(NSNumber.init)
    }
    func shapedType(_ shape: [Int]?) -> MPSGraphShapedType {
      MPSGraphShapedType(shape: nsShape(shape), dataType: dataType.mps)
    }
    func placeholder(_ shape: [Int]?, _ name: String) -> MPSGraphTensor {
      graph.placeholder(
        shape: nsShape(shape), dataType: dataType.mps, name: name)
    }
    func transpose(
      _ tensor: MPSGraphTensor, _ name: String,
      batchDims: [Int], permutation: [Int]
    ) -> MPSGraphTensor {
      let batchPart = Array<Int>(batchDims.indices.map { $0 })
      let permPart = permutation.map { $0 + batchDims.count }
      let _permutation = nsShape(batchPart + permPart)
      return graph.transpose(tensor, permutation: _permutation, name: name)
    }
    
    var originalQ: MPSGraphTensor
    var shapedTypeQ: MPSGraphShapedType
    var postTransposeQ: MPSGraphTensor
    if parameters.Q_trans {
      originalQ = placeholder(qShapeTranspose, "Q_trans")
      shapedTypeQ = shapedType(qShapeTranspose)
      postTransposeQ = transpose(
        originalQ, "Q", batchDims: qBatch, permutation: [2, 0, 1])
    } else {
      originalQ = placeholder(qShape, "Q")
      shapedTypeQ = shapedType(qShape)
      postTransposeQ = originalQ
    }
    
    var originalK: MPSGraphTensor
    var shapedTypeK: MPSGraphShapedType
    var postTransposeK: MPSGraphTensor
    if parameters.K_trans {
      originalK = placeholder(kShapeTranspose, "K_trans")
      shapedTypeK = shapedType(kShapeTranspose)
      postTransposeK = transpose(
        originalK, "K", batchDims: qBatch, permutation: [1, 2, 0])
    } else {
      originalK = placeholder(kShape, "K")
      shapedTypeK = shapedType(kShape)
      postTransposeK = originalK
    }
    
    var originalV: MPSGraphTensor
    var shapedTypeV: MPSGraphShapedType
    var postTransposeV: MPSGraphTensor
    if parameters.V_trans {
      originalV = placeholder(vShapeTranspose, "V_trans")
      shapedTypeV = shapedType(vShapeTranspose)
      postTransposeV = transpose(
        originalV, "V", batchDims: qBatch, permutation: [2, 0, 1])
    } else {
      originalV = placeholder(vShape, "V")
      shapedTypeV = shapedType(vShape)
      postTransposeV = originalV
    }
    
    var originalMask: MPSGraphTensor?
    var shapedTypeMask: MPSGraphShapedType?
    if let maskShape {
      originalMask = placeholder(maskShape, "mask")
      shapedTypeMask = shapedType(maskShape)
    }
    
    let contiguousQ = transpose(
      postTransposeQ, "Q_contiguous", batchDims: qBatch, permutation: [1, 0, 2])
    var attentionMatrix = graph.matrixMultiplication(
      primary: contiguousQ, secondary: postTransposeK, name: "QK")
    let alpha = graph.constant(
      rsqrt(Double(parameters.D)), dataType: dataType.mps)
    attentionMatrix = graph.multiplication(
      attentionMatrix, alpha, name: "QK/sqrt(D)")
    
    if let originalMask {
      attentionMatrix = graph.addition(
        attentionMatrix, originalMask, name: "mask(QK/sqrt(D))")
    }
    if dataType == .half {
      attentionMatrix = graph.cast(
        attentionMatrix, to: .float32, name: "QK_f32")
    }
    attentionMatrix = graph.softMax(
      with: attentionMatrix, axis: qShape.count - 1, name: "sm(QK)")
    
    let contiguousV = transpose(
      postTransposeV, "V_contiguous", batchDims: qBatch, permutation: [1, 0, 2])
    var contiguousO = graph.matrixMultiplication(
      primary: attentionMatrix, secondary: contiguousV, name: "O_contiguous")
    if dataType == .half {
      contiguousO = graph.cast(
        contiguousO, to: .float16, name: "O_contiguous_f16")
    }
    
    var originalO: MPSGraphTensor
    var shapedTypeO: MPSGraphShapedType
    var postTransposeO: MPSGraphTensor
    if parameters.O_trans {
      originalO = transpose(
        contiguousO, "O_trans", batchDims: qBatch, permutation: [1, 0, 2])
      shapedTypeO = shapedType(oShapeTranspose)
      postTransposeO = transpose(
        originalO, "O", batchDims: qBatch, permutation: [1, 2, 0])
    } else {
      originalO = transpose(
        contiguousO, "O", batchDims: qBatch, permutation: [1, 0, 2])
      shapedTypeO = shapedType(oShape)
      postTransposeO = originalO
    }
    var feeds: [MPSGraphTensor: MPSGraphShapedType] = [
      originalQ: shapedTypeQ,
      originalK: shapedTypeK,
      originalV: shapedTypeV,
      originalO: shapedTypeO,
    ]
    if let originalMask, let shapedTypeMask {
      feeds[originalMask] = shapedTypeMask
    }
    return AsyncGraph(
      graph: graph, feeds: feeds, targetTensors: [postTransposeO])
  }
}
