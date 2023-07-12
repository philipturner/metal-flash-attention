//
//  GEMM.swift
//  MetalFlashAttention
//
//  Created by Philip Turner on 6/27/23.
//

import Metal
import MetalPerformanceShadersGraph
import PythonKit

protocol GEMM: Operation {
  typealias Tensors = GEMM_Tensors
  var parameters: GEMM_Parameters { get set }

  init(parameters: GEMM_Parameters)
}

extension GEMM {
  func equals(_ other: GEMM) -> Bool {
    (type(of: self) == type(of: other)) && (parameters == other.parameters)
  }
}

struct GEMM_Parameters: Hashable, Equatable {
  var dataType: MTLDataType
  var M: Int
  var N: Int
  var K: Int
  var A_trans: Bool
  var B_trans: Bool
  var alpha: Float
  var beta: Float
  var batched: Bool
  var fused_activation: Bool
  
  // These are only needed by MPSGraph; MFA supports dynamic batch size.
  var batchDimensionsA: [Int]?
  var batchDimensionsB: [Int]?
}

struct GEMM_Tensors {
  var a: TensorBuffer
  var b: TensorBuffer
  var c: TensorBuffer
}

struct MFA_GEMM: GEMM, MFA_Operation {
  var parameters: GEMM_Parameters
  
  static var functionConstants: [String: MTLConvertible] = [
    "M_simd": UInt16(16), // 16-24
    "N_simd": UInt16(16), // 16-24
    "K_simd": UInt16(32), // 24-32
    "M_splits": UInt16(2), // 2
    "N_splits": UInt16(2), // 2
    "K_splits": UInt16(1), // 1-3
  ]
  
  init(parameters: GEMM_Parameters) {
    self.parameters = parameters
  }
  
  func makeAsyncResource() -> AsyncPipeline {
    let dataType = parameters.dataType
    precondition(dataType == .float || dataType == .half)
    precondition(parameters.alpha == 1.0)
    precondition(parameters.beta == 0.0)
    precondition(parameters.fused_activation == false)
    
    let constants = MTLFunctionConstantValues()
    var pcopy = self.parameters
    constants.setConstantValue(&pcopy.M, type: .uint, index: 0)
    constants.setConstantValue(&pcopy.N, type: .uint, index: 1)
    constants.setConstantValue(&pcopy.K, type: .uint, index: 2)
    constants.setConstantValue(&pcopy.A_trans, type: .bool, index: 10)
    constants.setConstantValue(&pcopy.B_trans, type: .bool, index: 11)
    constants.setConstantValue(&pcopy.alpha, type: .float, index: 20)
    constants.setConstantValue(&pcopy.beta, type: .float, index: 21)
    constants.setConstantValue(&pcopy.batched, type: .bool, index: 100)
    constants.setConstantValue(&pcopy.fused_activation, type: .bool, index: 101)
    
    var M_simd = Self.functionConstants["M_simd"] as! UInt16
    var N_simd = Self.functionConstants["N_simd"] as! UInt16
    var K_simd = Self.functionConstants["K_simd"] as! UInt16
    var M_splits = Self.functionConstants["M_splits"] as! UInt16
    var N_splits = Self.functionConstants["N_splits"] as! UInt16
    var K_splits = Self.functionConstants["K_splits"] as! UInt16
    constants.setConstantValue(&M_simd, type: .ushort, index: 200)
    constants.setConstantValue(&N_simd, type: .ushort, index: 201)
    constants.setConstantValue(&K_simd, type: .ushort, index: 202)
    constants.setConstantValue(&M_splits, type: .ushort, index: 210)
    constants.setConstantValue(&N_splits, type: .ushort, index: 211)
    constants.setConstantValue(&K_splits, type: .ushort, index: 212)
    
    var name: String
    switch dataType {
    case .half: name = "hgemm"
    case .float: name = "sgemm"
    default: fatalError()
    }
    
    let library = MetalContext.global.library
    let function = try! library.makeFunction(
      name: name, constantValues: constants)
    
    let M_group = M_simd * M_splits
    let N_group = N_simd * N_splits
    let K_group = K_simd * K_splits
    let A_block_length = M_group * K_group
    let B_block_length = K_group * N_group
    
    var block_elements = A_block_length + B_block_length;
    if (pcopy.M % 8 != 0) && (pcopy.N % 8 != 0) {
      let C_block_length = M_group * N_group;
      block_elements = max(C_block_length, block_elements)
    }
    var blockBytes = block_elements * UInt16(dataType.size)
    
    func ceilDivide(target: Int, granularity: UInt16) -> Int {
      (target + Int(granularity) - 1) / Int(granularity)
    }
    let gridSize = MTLSize(
      width: ceilDivide(target: parameters.N, granularity: N_group),
      height: ceilDivide(target: parameters.M, granularity: M_group),
      depth: 1)
    let groupSize = MTLSize(
      width: 128 * Int(K_splits),
      height: 1,
      depth: 1)
    
    return AsyncPipeline(
      function: function,
      batched: parameters.batched,
      threadgroupMemoryLength: blockBytes,
      gridSize: gridSize,
      groupSize: groupSize)
  }
  
  func encode(
    encoder: MTLComputeCommandEncoder,
    tensors: GEMM_Tensors,
    resource: AsyncPipeline
  ) {
    encoder.setComputePipelineState(resource.resource)
    encoder.setThreadgroupMemoryLength(
      Int(resource.threadgroupMemoryLength), index: 0)
    
    let tensorA = tensors.a as! MFA_TensorBuffer
    let tensorB = tensors.b as! MFA_TensorBuffer
    let tensorC = tensors.c as! MFA_TensorBuffer
    encoder.setBuffer(tensorA.buffer, offset: 0, index: 0)
    encoder.setBuffer(tensorB.buffer, offset: 0, index: 1)
    encoder.setBuffer(tensorC.buffer, offset: 0, index: 2)
    
    var gridZ: Int
    if resource.batched {
      let batchDimensionsA = tensors.a.shape.dropLast(2)
      let batchDimensionsB = tensors.b.shape.dropLast(2)
      let batchDimensionsC = tensors.c.shape.dropLast(2)
      assert(batchDimensionsA.reduce(1, *) > 0)
      assert(
        batchDimensionsB.reduce(1, *) == 1 ||
        batchDimensionsB == batchDimensionsA)
      assert(batchDimensionsA == batchDimensionsC)
      gridZ = batchDimensionsA.reduce(1, *)
      
      // Mixed precision will cause undefined behavior.
      let elementSize = tensors.a.dataType.size
      func byteStride(shape: [Int]) -> Int {
        let rank = shape.count
        var output = elementSize * shape[rank - 2] * shape[rank - 1]
        if shape.dropLast(2).reduce(1, *) == 1 {
          output = 0
        }
        return output
      }
      let byteStrideA = byteStride(shape: tensors.a.shape)
      let byteStrideB = byteStride(shape: tensors.b.shape)
      let byteStrideC = byteStride(shape: tensors.c.shape)
      withUnsafeTemporaryAllocation(
        of: SIMD3<UInt64>.self, capacity: gridZ
      ) { buffer in
        for i in 0..<buffer.count {
          buffer[i] = SIMD3(
            UInt64(truncatingIfNeeded: i * byteStrideA),
            UInt64(truncatingIfNeeded: i * byteStrideB),
            UInt64(truncatingIfNeeded: i * byteStrideC))
        }
        
        let bufferLength = buffer.count * MemoryLayout<SIMD3<UInt64>>.stride
        assert(MemoryLayout<SIMD3<UInt64>>.stride == 8 * 4)
        encoder.setBytes(buffer.baseAddress!, length: bufferLength, index: 10)
      }
    } else {
      assert(tensors.a.shape.count == 2)
      assert(tensors.b.shape.count == 2)
      assert(tensors.c.shape.count == 2)
      gridZ = 1
    }
    
    var gridSize = resource.gridSize
    gridSize.depth = gridZ
    encoder.dispatchThreadgroups(
      gridSize, threadsPerThreadgroup: resource.groupSize)
  }
}

struct MPS_GEMM: GEMM, MPS_Operation {
  var parameters: GEMM_Parameters
  
  init(parameters: GEMM_Parameters) {
    self.parameters = parameters
  }
  
  func makeAsyncResource() -> AsyncGraph {
    let dataType = parameters.dataType
    precondition(dataType == .float || dataType == .half)
    precondition(parameters.alpha == 1.0)
    precondition(parameters.beta == 0.0)
    precondition(parameters.fused_activation == false)
    if parameters.batched {
      precondition(parameters.batchDimensionsA!.count > 0)
      precondition(parameters.batchDimensionsA!.reduce(1, *) > 0)
      if let batchDimensionsB = parameters.batchDimensionsB {
        precondition(
          batchDimensionsB.reduce(1, *) == 1 ||
          batchDimensionsB == parameters.batchDimensionsA!)
      }
    } else {
      precondition(parameters.batchDimensionsA! == [])
      precondition(parameters.batchDimensionsB! == [])
    }
    
    let aBatch: [Int] = parameters.batchDimensionsA!
    let bBatch: [Int] = parameters.batchDimensionsB!
    let aShape: [Int] = aBatch + [parameters.M, parameters.K]
    let bShape: [Int] = bBatch + [parameters.K, parameters.N]
    
    var aShapeTranspose: [Int]?
    var bShapeTranspose: [Int]?
    if parameters.A_trans {
      aShapeTranspose = aBatch + [parameters.K, parameters.M]
    }
    if parameters.B_trans {
      bShapeTranspose = bBatch + [parameters.N, parameters.K]
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
      _ tensor: MPSGraphTensor, _ name: String, batchDims: [Int]
    ) -> MPSGraphTensor {
      graph.transposeTensor(
        tensor,
        dimension: batchDims.count + 0,
        withDimension: batchDims.count + 1,
        name: name)
    }
    
    var originalA: MPSGraphTensor
    var shapedTypeA: MPSGraphShapedType
    var postTransposeA: MPSGraphTensor
    if parameters.A_trans {
      originalA = placeholder(aShapeTranspose, "A_trans")
      shapedTypeA = shapedType(aShapeTranspose)
      postTransposeA = transpose(originalA, "A", batchDims: aBatch)
    } else {
      originalA = placeholder(aShape, "A_trans")
      shapedTypeA = shapedType(aShape)
      postTransposeA = originalA
    }
    
    var originalB: MPSGraphTensor
    var shapedTypeB: MPSGraphShapedType
    var postTransposeB: MPSGraphTensor
    if parameters.B_trans {
      originalB = placeholder(bShapeTranspose, "B_trans")
      shapedTypeB = shapedType(bShapeTranspose)
      postTransposeB = transpose(originalB, "B", batchDims: bBatch)
    } else {
      originalB = placeholder(bShape, "B_trans")
      shapedTypeB = shapedType(bShape)
      postTransposeB = originalB
    }
    
    let tensorC = graph.matrixMultiplication(
      primary: postTransposeA, secondary: postTransposeB, name: "C")
    return AsyncGraph(
      graph: graph,
      feeds: [originalA: shapedTypeA, originalB: shapedTypeB],
      targetTensors: [tensorC])
  }
  
  func encode(
    encoder: MPSCommandBuffer,
    tensors: GEMM_Tensors,
    resource: AsyncGraph
  ) {
    let tensorA = tensors.a as! MPS_TensorBuffer
    let tensorB = tensors.b as! MPS_TensorBuffer
    let tensorC = tensors.c as! MPS_TensorBuffer
    let inputs = [tensorA.tensorData, tensorB.tensorData]
    let results = [tensorC.tensorData]
    resource.resource.encode(
      to: encoder,
      inputs: inputs,
      results: results,
      executionDescriptor: nil)
  }
}

struct Py_GEMM: GEMM, Py_Operation {
  var parameters: GEMM_Parameters
  
  init(parameters: GEMM_Parameters) {
    self.parameters = parameters
  }
  
  func execute(tensors: GEMM_Tensors) {
    let tensorA = tensors.a as! Py_TensorBuffer
    let tensorB = tensors.b as! Py_TensorBuffer
    let tensorC = tensors.c as! Py_TensorBuffer
  
    let np = PythonContext.global.np
    if !parameters.A_trans && !parameters.B_trans {
      np.matmul(tensorA.ndarray, tensorB.ndarray, out: tensorC.ndarray)
      // repr = "ij,jk->ik"
      // repr = "ik,kj->ij"
    } else {
      var repr: String?
      if parameters.batched {
        assert(parameters.batchDimensionsA!.reduce(1, *) > 0)
        assert(
          parameters.batchDimensionsB!.reduce(1, *) == 1 ||
          parameters.batchDimensionsB! == parameters.batchDimensionsA!)
        
        
        var extraDimsRepr: String = ""
        if parameters.batchDimensionsA!.count >= 2 {
          if parameters.batchDimensionsA!.count >= 3 {
            fatalError("Case not supported yet.")
          }
          extraDimsRepr = "g"
        }
        
        var bInsert: String = ""
        if parameters.batchDimensionsB!.count > 0 {
          if parameters.batchDimensionsB!.reduce(1, *) == 1 {
            fatalError("Case not supported yet.")
          }
          precondition(
            parameters.batchDimensionsB! == parameters.batchDimensionsA!)
          bInsert = "\(extraDimsRepr)h"
        }
        extraDimsRepr += "h"
        if parameters.A_trans && parameters.B_trans {
          repr = "\(extraDimsRepr)ji,\(bInsert)kj->\(extraDimsRepr)ik"
        } else if parameters.A_trans {
          repr = "\(extraDimsRepr)ji,\(bInsert)jk->\(extraDimsRepr)ik"
        } else if parameters.B_trans {
          repr = "\(extraDimsRepr)ij,\(bInsert)kj->\(extraDimsRepr)ik"
        }
      } else {
        if parameters.A_trans && parameters.B_trans {
          repr = "ji,kj->ik"
        } else if parameters.A_trans {
          repr = "ji,jk->ik"
        } else if parameters.B_trans {
          repr = "ij,kj->ik"
        }
      }
      np.einsum(repr!, tensorA.ndarray, tensorB.ndarray, out: tensorC.ndarray)
    }
  }
}
