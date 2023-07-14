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
  typealias Tensors = Attention_Tensors
  var parameters: Attention_Parameters { get set }
  
  init(parameters: Attention_Parameters)
}

extension Attention {
  func equals(_ other: Attention) -> Bool {
    (type(of: self) == type(of: other)) && (parameters == other.parameters)
  }
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

struct MFA_Attention: Attention, MFA_Operation {
  var parameters: Attention_Parameters
  
  static var functionConstants: [String: MTLConvertible] = [
    "R_simd": UInt16(16),
    "C_simd": UInt16(64),
    "R_splits": UInt16(4),
  ]
  
  init(parameters: Attention_Parameters) {
    self.parameters = parameters
  }
  
  func makeAsyncResource() -> AsyncPipeline {
    let dataType = parameters.dataType
    precondition(dataType == .float || dataType == .half)
    precondition(!parameters.block_sparse, "Block sparsity not supported yet.")
    
    let constants = MTLFunctionConstantValues()
    var pcopy = self.parameters
    constants.setConstantValue(&pcopy.R, type: .uint, index: 0)
    constants.setConstantValue(&pcopy.C, type: .uint, index: 1)
    constants.setConstantValue(&pcopy.H, type: .uint, index: 2)
    constants.setConstantValue(&pcopy.D, type: .uint, index: 3)
    constants.setConstantValue(&pcopy.Q_trans, type: .bool, index: 10)
    constants.setConstantValue(&pcopy.K_trans, type: .bool, index: 11)
    constants.setConstantValue(&pcopy.V_trans, type: .bool, index: 12)
    constants.setConstantValue(&pcopy.O_trans, type: .bool, index: 13)
    
    var dataTypeRawValue = dataType.rawValue
    constants.setConstantValue(&dataTypeRawValue, type: .uint, index: 30)
    constants.setConstantValue(&pcopy.batched, type: .bool, index: 50000)
    constants.setConstantValue(&pcopy.block_sparse, type: .bool, index: 102)
    
    var forward = true
    var backward = false
    var generateBlockMask = false
    constants.setConstantValue(&forward, type: .bool, index: 103)
    constants.setConstantValue(&backward, type: .bool, index: 104)
    constants.setConstantValue(&generateBlockMask, type: .bool, index: 105)
    
    var R_simd = Self.functionConstants["R_simd"] as! UInt16
    var C_simd = Self.functionConstants["C_simd"] as! UInt16
    var R_splits = Self.functionConstants["R_splits"] as! UInt16
    constants.setConstantValue(&R_simd, type: .ushort, index: 200)
    constants.setConstantValue(&C_simd, type: .ushort, index: 201)
    constants.setConstantValue(&R_splits, type: .ushort, index: 210)
    
    let library = MetalContext.global.library
    let function = try! library.makeFunction(
      name: "attention", constantValues: constants)
    
    let D_simd = UInt16(pcopy.D + 7) / 8 * 8
    let R_group = R_simd * R_splits
    let Q_block_length = R_group * D_simd
    let K_block_length = D_simd * C_simd
    let V_block_length = C_simd * D_simd
    let O_block_length = R_group * D_simd
    
    var blockElements = max(Q_block_length, K_block_length)
    blockElements = max(blockElements, V_block_length)
    blockElements = max(blockElements, O_block_length)
    if pcopy.masked {
      let mask_block_length = R_group * C_simd
      blockElements = max(blockElements, mask_block_length)
    }
    let blockBytes = blockElements * UInt16(dataType.size)
    
    func ceilDivide(target: Int, granularity: UInt16) -> Int {
      (target + Int(granularity) - 1) / Int(granularity)
    }
    let gridSize = MTLSize(
      width: ceilDivide(target: parameters.R, granularity: R_group),
      height: parameters.H,
      depth: 1)
    let groupSize = MTLSize(
      width: 32 * Int(R_splits),
      height: 1,
      depth: 1)
    
    var flags: UInt32 = 0
    if parameters.batched {
      flags |= 0x1
    }
    if parameters.masked {
      flags |= 0x2
    }
    if parameters.block_sparse {
      flags |= 0x4
    }
    return AsyncPipeline(
      function: function,
      flags: flags,
      threadgroupMemoryLength: blockBytes,
      gridSize: gridSize,
      groupSize: groupSize)
  }
  
  func encode(
    encoder: MTLComputeCommandEncoder,
    tensors: Attention_Tensors,
    resource: AsyncPipeline
  ) {
    encoder.setComputePipelineState(resource.resource)
    encoder.setThreadgroupMemoryLength(
      Int(resource.threadgroupMemoryLength), index: 0)
    
    let tensorQ = tensors.q as! MFA_TensorBuffer
    let tensorK = tensors.k as! MFA_TensorBuffer
    let tensorV = tensors.v as! MFA_TensorBuffer
    let tensorO = tensors.o as! MFA_TensorBuffer
    encoder.setBuffer(tensorQ.buffer, offset: 0, index: 0)
    encoder.setBuffer(tensorK.buffer, offset: 0, index: 1)
    encoder.setBuffer(tensorV.buffer, offset: 0, index: 2)
    encoder.setBuffer(tensorO.buffer, offset: 0, index: 3)
    
    var gridZ: Int
    if resource.flags & 0x1 > 0 {
      let batchDimensionsQ = tensors.q.shape.dropLast(3)
      let batchDimensionsK = tensors.k.shape.dropLast(3)
      let batchDimensionsV = tensors.v.shape.dropLast(3)
      let batchDimensionsO = tensors.o.shape.dropLast(3)
      assert(batchDimensionsQ.reduce(1, *) > 0)
      assert(batchDimensionsQ == batchDimensionsK)
      assert(batchDimensionsQ == batchDimensionsV)
      assert(batchDimensionsQ == batchDimensionsO)
      gridZ = batchDimensionsQ.reduce(1, *)
      
      let elementSize = tensors.q.dataType.size
      func byteStride(shape: [Int]) -> Int {
        var output = elementSize
        output *= shape[shape.count - 1]
        output *= shape[shape.count - 2]
        output *= shape[shape.count - 3]
        if shape.dropLast(3).reduce(1, *) == 1 {
          output = 0
        }
        return output
      }
      var byteStrideMask = 0
      let byteStrideBlockMask = 0
      
      if resource.flags & 0x2 > 0 {
        let batchDimensionsMask = tensors.mask!.shape.dropLast(3)
        assert(
          batchDimensionsMask.reduce(1, *) == 1 ||
          batchDimensionsMask == batchDimensionsQ)
        byteStrideMask = byteStride(shape: tensors.mask!.shape)
      }
      
      withUnsafeTemporaryAllocation(
        of: SIMD4<UInt64>.self, capacity: gridZ
      ) { buffer in
        for i in 0..<buffer.count {
          buffer[i] = SIMD4(
            UInt64(truncatingIfNeeded: i * byteStrideMask),
            UInt64(truncatingIfNeeded: i * byteStrideBlockMask),
            UInt64(0),
            UInt64(0))
        }
        
        let bufferLength = buffer.count * MemoryLayout<SIMD4<UInt64>>.stride
        assert(MemoryLayout<SIMD4<UInt64>>.stride == 8 * 4)
        encoder.setBytes(buffer.baseAddress!, length: bufferLength, index: 10)
      }
    } else {
      assert(tensors.q.shape.count == 3)
      assert(tensors.k.shape.count == 3)
      assert(tensors.v.shape.count == 3)
      assert(tensors.o.shape.count == 3)
      if let tensorMask = tensors.mask {
        assert(tensorMask.shape.count == 3)
      }
      gridZ = 1
    }
    
    if resource.flags & 0x2 > 0 {
      let tensorMask = tensors.mask! as! MFA_TensorBuffer
      let maskShape = tensors.mask!.shape
      assert(maskShape[maskShape.count - 3] == 1)
      encoder.setBuffer(tensorMask.buffer, offset: 0, index: 11)
    }
    
    var gridSize = resource.gridSize
    gridSize.depth = gridZ
    encoder.dispatchThreadgroups(
      gridSize, threadsPerThreadgroup: resource.groupSize)
  }
}

struct MPS_Attention: Attention, MPS_Operation {
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
  
  func encode(
    encoder: MPSCommandBuffer,
    tensors: Attention_Tensors,
    resource: AsyncGraph
  ) {
    let tensorQ = tensors.q as! MPS_TensorBuffer
    let tensorK = tensors.k as! MPS_TensorBuffer
    let tensorV = tensors.v as! MPS_TensorBuffer
    let tensorO = tensors.o as! MPS_TensorBuffer
    var inputs = [tensorQ.tensorData, tensorK.tensorData, tensorV.tensorData]
    
    if let mask = tensors.mask {
      let tensorMask = mask as! MPS_TensorBuffer
      inputs.append(tensorMask.tensorData)
    }
    let results = [tensorO.tensorData]
    resource.resource.encode(
      to: encoder,
      inputs: inputs,
      results: results,
      executionDescriptor: nil)
  }
}

struct Py_Attention: Attention, Py_Operation {
  var parameters: Attention_Parameters
  
  init(parameters: Attention_Parameters) {
    self.parameters = parameters
  }
  
  func execute(tensors: Attention_Tensors) {
    let tensorQ = tensors.q as! Py_TensorBuffer
    let tensorK = tensors.k as! Py_TensorBuffer
    let tensorV = tensors.v as! Py_TensorBuffer
    let tensorO = tensors.o as! Py_TensorBuffer
    var tensorMask: Py_TensorBuffer?
    if parameters.masked {
      // WARNING: Mask dimensions need to be [B, 1, R, C].
      tensorMask = (tensors.mask! as! Py_TensorBuffer)
    }
    
    let np = PythonContext.global.np
    
    var postTransposeQ = tensorQ.ndarray
    if parameters.Q_trans {
      postTransposeQ = np.einsum("...ijk->...kij", tensorQ.ndarray)
    }
    
    var postTransposeK = tensorK.ndarray
    if parameters.K_trans {
      postTransposeK = np.einsum("...ijk->...jki", tensorK.ndarray)
    }
    
    var postTransposeV = tensorV.ndarray
    if parameters.V_trans {
      postTransposeV = np.einsum("...ijk->...kij", tensorV.ndarray)
    }
    
    // Multiply Q * K.
    // [R, H, D] * [H, D, C] -> [H, R, C]
    var attentionMatrix = np.einsum(
      "...ijk,...jkl->...jil", postTransposeQ, postTransposeK)
    attentionMatrix *= PythonObject(rsqrt(Double(parameters.D)))
    
    // Apply explicit mask.
    if let tensorMask {
      np.add(attentionMatrix, tensorMask.ndarray, out: attentionMatrix)
    }
    
    // Perform softmax.
    let lastAxis = PythonObject(tensorQ.shape.count - 1)
    let summary = np[dynamicMember: "max"](
      attentionMatrix, axis: lastAxis, keepdims: true)
    np.subtract(attentionMatrix, summary, out: attentionMatrix)
    np.exp(attentionMatrix, out: attentionMatrix)
    np.sum(
      attentionMatrix, axis: lastAxis, keepdims: true, out: summary)
    np.divide(attentionMatrix, summary, out: summary)
    
    // Multiply P * V.
    // [H, R, C] * [C, H, D] -> [R, H, D]
    let originalO = tensorO.ndarray
    np.einsum(
      "...ijk,...kil->jil", attentionMatrix, postTransposeV, out: originalO)
    if parameters.O_trans {
      np.einsum("...ijk->...jki", originalO, out: originalO)
    }
  }
}
