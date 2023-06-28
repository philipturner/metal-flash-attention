//
//  EuclideanDistance.swift
//  MetalFlashAttention
//
//  Created by Philip Turner on 6/27/23.
//

import Accelerate
import Metal

struct EuclideanDistanceParameters {
  // `averageMagnitude` is 0.5 for uniformly distributed random numbers.
  var averageMagnitude: Float
  
  // `averageDeviation` is sqrt(K) during a matrix multiplication.
  var averageDeviation: Float
  
  init(averageMagnitude: Float, averageDeviation: Float) {
    self.averageMagnitude = averageMagnitude
    self.averageDeviation = averageDeviation
  }
  
  init(matrixK: Int) {
    self.averageMagnitude = 0.5 * Float(matrixK)
    self.averageDeviation = sqrt(averageMagnitude)
  }
}

extension Tensor {
  func euclideanDistance(to other: Tensor<Element>) -> Float {
    buffer.euclideanDistance(to: other.buffer)
  }
  
  func isApproximatelyEqual(
    to other: Tensor<Element>,
    parameters: EuclideanDistanceParameters
  ) -> Bool {
    precondition(self.count == other.count)
    var tolerance = Float(self.count)
    
    let averageMagnitude = parameters.averageMagnitude
    let averageDeviation = parameters.averageDeviation
    switch Element.mtlDataType {
    case .float:
      tolerance *= max(0.002 * averageMagnitude, 3e-7 * averageDeviation)
    case .half:
      tolerance *= max(0.02 * averageMagnitude, 1e-2 * averageDeviation)
    default: fatalError()
    }
    return euclideanDistance(to: other) < tolerance
  }
}

extension TensorBuffer {
  func euclideanDistance(to other: TensorBuffer) -> Float {
    precondition(self.dataType == other.dataType)
    precondition(self.count == other.count)
    
    let x_f32: UnsafeMutablePointer<Float> = .allocate(capacity: self.count)
    let y_f32: UnsafeMutablePointer<Float> = .allocate(capacity: other.count)
    defer { x_f32.deallocate() }
    defer { y_f32.deallocate() }
    
    if dataType == .half {
      // Partially sourced from:
      // https://github.com/hollance/TensorFlow-iOS-Example/blob/master/VoiceMetal/VoiceMetal/Float16.swift
      func copy(dst: UnsafeMutableRawPointer, src: UnsafeMutableRawPointer) {
        let count = self.count
        var bufferFloat16 = vImage_Buffer(
          data: src, height: 1, width: UInt(count), rowBytes: count * 2)
        var bufferFloat32 = vImage_Buffer(
          data: dst, height: 1, width: UInt(count), rowBytes: count * 4)
        
        let error = vImageConvert_Planar16FtoPlanarF(
          &bufferFloat16, &bufferFloat32, 0)
        if error != kvImageNoError {
          fatalError(
            "Encountered error code \(error) while converting F16 to F32.")
        }
      }
      copy(dst: x_f32, src: self.pointer)
      copy(dst: y_f32, src: other.pointer)
    } else {
      memcpy(x_f32, self.pointer, self.allocatedSize)
      memcpy(y_f32, other.pointer, other.allocatedSize)
    }
    
    var difference = [Float](repeating: 0, count: count)
    memcpy(&difference, x_f32, count * 4)
    var n_copy = Int32(count)
    var a = Float(-1)
    var inc = Int32(1)
    var inc_copy = inc
    
    // Find x + (-1 * y)
    saxpy_(&n_copy, &a, y_f32, &inc, &difference, &inc_copy)
    
    // Find ||x - y||
    return Float(snrm2_(&n_copy, &difference, &inc))
  }
}
