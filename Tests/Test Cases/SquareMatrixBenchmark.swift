//
//  SquareMatrixBenchmark.swift
//  MetalFlashAttention
//
//  Created by Philip Turner on 6/29/23.
//

import AppleGPUInfo
import Foundation
import PythonKit

func runSquareMatrixBenchmark() {
  let benchmark1 = SquareMatrixBenchmark<MFATestCase.Real>(
    range: 990..<1011, iterations: 1)
  benchmark1.profile(trials: 1)

  let benchmark20 = SquareMatrixBenchmark<MFATestCase.Real>(
    range: 990..<1011, iterations: 16)
  benchmark20.profile(trials: 8)
}

func SquareMatrixBenchmark_configure(
  _ iterations: inout Int, _ trials: inout Int
) {
  // Try to keep test time constant across devices.
  let cores = MetalContext.global.infoDevice.coreCount
  trials = cores / 4
  if trials > 16 {
    trials = 16
  } else if trials < 2 {
    if iterations >= 4 {
      iterations /= 2
    }
    trials = 2
  }
}

func SquareMatrixBenchmark_configure_2(
  _ iterations: inout Int, _ trials: inout Int, ref: Int
) {
  // Try to keep test time constant across devices.
  let cores = MetalContext.global.infoDevice.coreCount
  trials = cores / (32 / ref)
  if trials > 16 {
    trials = 16
  } else if trials < 2 {
    if iterations >= 4 {
      iterations /= 2
    }
    trials = 2
  }
}

struct SquareMatrixBenchmark<T: TensorElement> {
  var range: Range<Int>
  var iterations: Int
  var _trials: Int
  
  static func configure(iterations: inout Int, trials: inout Int) {
    // Try to keep test time constant across devices.
    let cores = MetalContext.global.infoDevice.coreCount
    trials = cores / 4
    if trials > 16 {
      trials = 16
    } else if trials < 2 {
      if iterations >= 4 {
        iterations /= 2
      }
      trials = 2
    }
  }
  
  init(range: Range<Int>, iterations: Int) {
    self.range = range
    self.iterations = iterations
    self._trials = 0
    SquareMatrixBenchmark_configure(&self.iterations, &self._trials)
  }
  
  func profile(trials: Int) {
    for matrixSize in range {
      let M: Int = matrixSize
      let N: Int = matrixSize
      let K: Int = matrixSize
      let params = EuclideanDistanceParameters(matrixK: K, batchSize: nil)
      
      let py_A = Tensor<T>(
        shape: [M, K], randomUniform: 0..<1, backend: .numpy)
      let py_B = Tensor<T>(
        shape: [K, N], randomUniform: 0..<1, backend: .numpy)
      var tensorsC: [TensorBackend: Tensor<T>] = [:]
      var minTimes: [TensorBackend: Double] = [:]
      
      for backend in [TensorBackend.mfa, .mps] {
        _ExecutionContext.defaultBackend = backend
        let A = Tensor<T>(copying: py_A)
        let B = Tensor<T>(copying: py_B)
        var C = Tensor<T>(zerosLike: [M, N])
        var minTime: Double = .infinity
        
        // Run for several trials.
        for _ in 0..<trials {
          let time = _ExecutionContext.profileCommands {
            for _ in 0..<iterations {
              C.matmul(A, B)
            }
          }
          minTime = min(time, minTime)
        }
        
        tensorsC[backend] = C
        precondition(minTimes[backend] == nil)
        minTimes[backend] = minTime
        _ExecutionContext.defaultBackend = .numpy
      }
      
      let tensorMFA = tensorsC[.mfa]!
      let tensorMPS = tensorsC[.mps]!
      let isEqual = tensorMFA.isApproximatelyEqual(
        to: tensorMPS, parameters: params)
      guard isEqual else {
        MPL_showComparison(
          actual: tensorMFA, expected: tensorMPS, parameters: params)
        fatalError("Tensors did not match.")
      }
      
      func utilizationRepr(_ time: Double) -> String {
        let ctx = MetalContext.global
        let floatOps = 2 * M * N * K * iterations
        let flops = Double(floatOps) / time
        let utilization = flops / ctx.infoDevice.flops
        let percent = utilization * 100
        return String(format: "%.1f", percent) + "%"
      }
      
      var output: String = "\(M)x\(N)x\(K)"
      if T.self == Float.self {
        output += "xf32 - "
      } else {
        output += "xf16 - "
      }
      output += "MFA \(utilizationRepr(minTimes[.mfa]!)) - "
      output += "MPS \(utilizationRepr(minTimes[.mps]!))"
      print(output)
    }
  }
}
