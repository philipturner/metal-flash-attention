//
//  CorrectnessTests.swift
//  MetalFlashAttention
//
//  Created by Philip Turner on 6/29/23.
//

import Metal
import QuartzCore
import PythonKit
import simd

class CorrectnessTests: MFATestCase {
  override class func typeDescription() -> String {
    "CorrectnessTests"
  }
  
  static let useBlockSparsity = true
  
  override func runQuickTests() {
    let logProgress = true
    testRandomAttention(logProgress: logProgress)
    testRandomGEMM(logProgress: logProgress)
  }
  
  func testRandomGEMM(logProgress: Bool) {
    print()
    let start = CACurrentMediaTime()
    
    //  0 - 15: batch 1, NN
    // 15 - 45: batch 1, NN/NT/TN/TT
    // 45 - 90: batch 2-16 for A/C
    // 90+: specific regression tests
    let numNonTransposedTrials = 15
    let numNonBatchedTrials = 45
    let nonBroadcastedCutoff = 60
    var numTrials = numNonBatchedTrials + 45
    
    // Create a biased random distribution that favors smaller numbers. Take the
    // uniform distribution, then cube the results.
    var allRandomInts: [SIMD3<Int>] = (0..<numTrials).map { i in
      let maxMatrixDimension = (i < numNonBatchedTrials) ? 1000 : 500
      
      var randomVecFloat = SIMD3<Float>.random(in: 0..<1)
      randomVecFloat = randomVecFloat * randomVecFloat * randomVecFloat
      var randomInts = SIMD3<Int>(randomVecFloat * Float(maxMatrixDimension))
      randomInts.replace(with: .one, where: randomInts .== .zero)
      return randomInts
    }
    allRandomInts.append(SIMD3(77, 64, 77))
    
    var allRandomTransposes: [(Bool, Bool, Bool)] = (0..<numTrials).map { i in
      if i < numNonTransposedTrials {
        return (false, false, false)
      } else {
        return (Bool.random(), Bool.random(), Bool.random())
      }
    }
    allRandomTransposes.append((false, false, false))
    
    var allRandomB: [(Int?, Int?)] = (0..<numTrials).map { i in
      if i < numNonBatchedTrials {
        return (nil, nil)
      } else {
        return (nil, Int.random(in: 2...16))
      }
    }
    allRandomB.append((2, 12))
    
    numTrials += 1
    
    func testRandomSize(index: Int, ghost: Bool) {
      let randomInts = allRandomInts[index]
      let randomTransposes = allRandomTransposes[index]
      let (extraDim, batchSize) = allRandomB[index]
      let useBias = index % 3 == 0
      
      let M = randomInts[0]
      let N = randomInts[1]
      let K = randomInts[2]
      let A_trans = randomTransposes.0
      let B_trans = randomTransposes.1
      let D_trans = randomTransposes.2
      let DTypeRepr = (Real.self == Float.self) ? "f32" : "f16"
      var transRepr = (A_trans ? "T" : "N") + (B_trans ? "T" : "N")
      if useBias {
        if D_trans {
          transRepr += "T, bias"
        } else {
          transRepr += "N, bias"
        }
      }
      
      var shapeA = A_trans ? [K, M] : [M, K]
      var shapeB = B_trans ? [N, K] : [K, N]
      var shapeC = [M, N]
      var shapeD = D_trans ? [M] : [N]
      if let batchSize {
        shapeA = [batchSize] + shapeA
        if index < nonBroadcastedCutoff {
          if index % 2 == 0 {
            shapeB = [1] + shapeB
            shapeD = [1] + shapeD
          }
        } else {
          shapeB = [batchSize] + shapeB
          shapeD = [batchSize] + shapeD
        }
        shapeC = [batchSize] + shapeC
      }
      if let extraDim {
        shapeA = [extraDim] + shapeA
        shapeB = [extraDim] + shapeB
        shapeC = [extraDim] + shapeC
        shapeD = [extraDim] + shapeD
      }
      
      let mps_A = Tensor<Real>(
        shape: shapeA, randomUniform: 0..<1, backend: .mps)
      let mps_B = Tensor<Real>(
        shape: shapeB, randomUniform: 0..<1, backend: .mps)
      var mps_C = Tensor<Real>(
        zerosLike: shapeC, backend: .mps)
      
      let mfa_A = Tensor(copying: mps_A, backend: .mfa)
      let mfa_B = Tensor(copying: mps_B, backend: .mfa)
      var mfa_C = Tensor(copying: mps_C, backend: .mfa)
      
      var mps_D: Tensor<Real>?
      var mfa_D: Tensor<Real>?
      if useBias {
        mps_D = Tensor<Real>(shape: shapeD, randomUniform: 0..<1, backend: .mps)
        mfa_D = Tensor(copying: mps_D!, backend: .mfa)
      }
      
      func act(
        A: Tensor<Real>,
        B: Tensor<Real>,
        C: inout Tensor<Real>,
        D: Tensor<Real>?) {
        C.matmul(
          A, B, D,
          transposeA: A_trans, transposeB: B_trans, transposeD: D_trans,
          fusedBias: D != nil)
      }
      
      if ghost {
        _ExecutionContext.withDefaultBackend(.mps) {
          TensorBackend.default.withGhostExecution {
            TensorBackend.default.markFirstCommand()
            act(A: mps_A, B: mps_B, C: &mps_C, D: mps_D)
            TensorBackend.default.markLastCommand()
            _ = TensorBackend.default.synchronize()
          }
        }
        _ExecutionContext.withDefaultBackend(.mfa) {
          TensorBackend.default.withGhostExecution {
            TensorBackend.default.markFirstCommand()
            act(A: mfa_A, B: mfa_B, C: &mfa_C, D: mfa_D)
            TensorBackend.default.markLastCommand()
            _ = TensorBackend.default.synchronize()
          }
        }
      } else {
        _ExecutionContext.withDefaultBackend(.mps) {
          TensorBackend.default.markFirstCommand()
          act(A: mps_A, B: mps_B, C: &mps_C, D: mps_D)
          TensorBackend.default.markLastCommand()
          _ = TensorBackend.default.synchronize()
        }
        _ExecutionContext.withDefaultBackend(.mfa) {
          TensorBackend.default.markFirstCommand()
          act(A: mfa_A, B: mfa_B, C: &mfa_C, D: mfa_D)
          TensorBackend.default.markLastCommand()
          _ = TensorBackend.default.synchronize()
        }
        
        let params = EuclideanDistanceParameters(
          matrixK: K, batchSize: batchSize)
        if !mfa_C.isApproximatelyEqual(to: mps_C, parameters: params) {
          do {
            var shapeRepr: String
            if let batchSize {
              shapeRepr = "\(batchSize)x\(M)x\(N)x\(K)x\(DTypeRepr)"
            } else {
              shapeRepr = "\(M)x\(N)x\(K)x\(DTypeRepr)"
            }
            if let extraDim {
              shapeRepr = "\(extraDim)x\(shapeRepr)"
            }
            let dist = mfa_C.euclideanDistance(to: mps_C)
            let distRepr = "- \(String(format: "%.3f", dist))"
            print("Failed test: \(shapeRepr) (\(transRepr)) \(distRepr)")
          }
          
          MPL_showComparison(
            actual: mfa_C, actualName: "MFA",
            expected: mps_C, expectedName: "MPS", parameters: params)
          fatalError("Tensors did not match.")
        }
        if logProgress {
          var shapeRepr: String
          if let batchSize {
            shapeRepr = "\(batchSize)x\(M)x\(N)x\(K)x\(DTypeRepr)"
          } else {
            shapeRepr = "\(M)x\(N)x\(K)x\(DTypeRepr)"
          }
          if let extraDim {
            shapeRepr = "\(extraDim)x\(shapeRepr)"
          }
          let dist = mfa_C.euclideanDistance(to: mps_C)
          let distRepr = "- \(String(format: "%.3f", dist))"
          print("Passed test: \(shapeRepr) (\(transRepr)) \(distRepr)")
        }
      }
      mps_A.buffer.release()
      mps_B.buffer.release()
      mps_C.buffer.release()
      mps_D?.buffer.release()
    }
    
    for i in 0..<numTrials {
      autoreleasepool {
        testRandomSize(index: i, ghost: true)
      }
    }
    for i in 0..<numTrials {
      autoreleasepool {
        testRandomSize(index: i, ghost: false)
      }
    }
    
    let end = CACurrentMediaTime()
    let repr = String(format: "%.3f", end - start)
    print("Finished 'testRandomGEMM' in \(repr) seconds.")
  }
  
  func testRandomAttention(logProgress: Bool) {
    print()
    let start = CACurrentMediaTime()
    
    let testExtension: Float = 0.334
    
    //  0 - 15: batch 1, K^T
    // 15 - 30: batch 1, all transposes
    // 30 - 45: batch 2-8
    // 45 - 60: batch 2-8, triangular mask
    // 60 - 75: batch 2-8, block-sparse mask, not matching block size
    // 75 - 90: batch 2-8, block-sparse mask, matching block size
    let numNonTransposedTrials = Int(15 * testExtension)
    let numNonBatchedTrials = Int(30 * testExtension)
    let triangularMaskedStart = Int(45 * testExtension)
    let unalignedSparseStart = Int(60 * testExtension)
    let alignedSparseStart = unalignedSparseStart + Int(15 * testExtension)
    let numTrials = alignedSparseStart + Int(15 * testExtension)
    
    // Create a biased random distribution that favors smaller numbers. Take the
    // uniform distribution, then cube the results.
    let allRandomInts: [SIMD4<Int>] = (0..<numTrials).map { i in
      let maxMatrixDimension = (i < numNonBatchedTrials) ? 515 : 259
      
      var randomVecFloat = SIMD4<Float>.random(in: 0..<1)
      precondition(all((randomVecFloat .>= 0) .| (randomVecFloat .<= 1)))
      
      var randomInts: SIMD4<Int>
      if i >= alignedSparseStart {
        let maxDim = Float(maxMatrixDimension)
        let floats = SIMD4(
          simd_mix(128, maxDim, randomVecFloat[0]),
          simd_mix(128, maxDim, randomVecFloat[1]),
          simd_mix(1, 3, randomVecFloat[2]),
          simd_mix(100, 128, randomVecFloat[3]))
        randomInts = SIMD4(__tg_rint(floats))
      } else {
        randomVecFloat = randomVecFloat * randomVecFloat * randomVecFloat
        if i >= triangularMaskedStart && i < unalignedSparseStart {
          let rms = sqrt(randomVecFloat[0] * randomVecFloat[1])
          randomVecFloat[0] = rms
          randomVecFloat[1] = rms
        }
        
        randomVecFloat *= Float(maxMatrixDimension)
        let numHeads =  Int.random(in: Bool.random() ? 1...2 : 3...8)
        randomVecFloat[2] /= Float(numHeads)
        
        randomInts = SIMD4(
          Int(randomVecFloat[0]),
          Int(randomVecFloat[1]),
          numHeads,
          Int(randomVecFloat[2]))
      }
      
      // WARNING: This threshold must change to match the block sizes.
      let threshold = (Real.self == Float.self) ? 128 : 256
      while randomInts[3] > threshold {
        randomInts[3] /= 2
        randomInts[2] *= 2
      }
      randomInts[2] = min(randomInts[2], 8)
      randomInts.replace(with: .one, where: randomInts .== .zero)
      return randomInts
    }
    
    let allRandomTransposes: [SIMD4<UInt8>] = (0..<numTrials).map { i in
      if i < numNonTransposedTrials {
        return SIMD4(0, 1, 0, 0)
      } else {
        let mask = UInt8.random(in: 0b0000...0b1111)
        return SIMD4(mask >> 0, mask >> 1, mask >> 2, mask >> 3) & 0b1
      }
    }
    
    let allRandomB: [Int?] = (0..<numTrials).map { i in
      if i < numNonBatchedTrials {
        return nil
      } else {
        return Int.random(in: 2...8)
      }
    }
    
    let allRandomMasks: [AttentionMask?] = (0..<numTrials).map { i in
      if i < triangularMaskedStart {
        return nil
      } else if i < unalignedSparseStart {
        return .upperTriangular
      } else {
        var blockSize: Int
        switch Int.random(in: 0..<3) {
        case 0:
          blockSize = Int.random(in: 1...5)
        case 1:
          blockSize = Int.random(in: 6...20)
        case 2:
          blockSize = Int.random(in: 21...100)
        default:
          fatalError()
        }
        if i >= alignedSparseStart {
          blockSize = 64
        }
        return .blockSparse(blockSize, Float.random(in: 0.1...0.9))
      }
    }
    
    func testRandomSize(index: Int, ghost: Bool) {
      let randomInts = allRandomInts[index]
      let randomTransposes = allRandomTransposes[index]
      let batchSize = allRandomB[index]
      let randomMask = allRandomMasks[index]
      
      let R = randomInts[0]
      let C = randomInts[1]
      let H = randomInts[2]
      let D = randomInts[3]
      
      let Q_trans = randomTransposes[0] == 1
      let K_trans = randomTransposes[1] == 1
      let V_trans = randomTransposes[2] == 1
      let O_trans = randomTransposes[3] == 1
      
      let DTypeRepr = (Real.self == Float.self) ? "f32" : "f16"
      let transRepr = [Q_trans, K_trans, V_trans, O_trans].reduce("") {
        $0 + ($1 ? "T" : "N")
      }
      
      var shapeQ = Q_trans ? [H, D, R] : [R, H, D]
      var shapeK = K_trans ? [C, H, D] : [H, D, C]
      var shapeV = V_trans ? [H, D, C] : [C, H, D]
      var shapeO = O_trans ? [H, D, R] : [R, H, D]
      var shapeMask = [1, R, C]
      
      if let batchSize {
        shapeQ = [batchSize] + shapeQ
        shapeK = [batchSize] + shapeK
        shapeV = [batchSize] + shapeV
        shapeO = [batchSize] + shapeO
        if index % 2 == 0 {
          shapeMask = [1] + shapeMask
        } else {
          shapeMask = [batchSize] + shapeMask
        }
      }
      
      let mps_Q = Tensor<Real>(
        shape: shapeQ, randomUniform: 0..<1, backend: .mps)
      let mps_K = Tensor<Real>(
        shape: shapeK, randomUniform: 0..<1, backend: .mps)
      let mps_V = Tensor<Real>(
        shape: shapeV, randomUniform: 0..<1, backend: .mps)
      var mps_O = Tensor<Real>(
        zerosLike: shapeO, backend: .mps)
      var mps_mask: Tensor<Real>?
      if let randomMask {
        mps_mask = Tensor<Real>(
          shape: shapeMask, mask: randomMask, backend: .mps)
      }
      
      let mfa_Q = Tensor(copying: mps_Q, backend: .mfa)
      let mfa_K = Tensor(copying: mps_K, backend: .mfa)
      let mfa_V = Tensor(copying: mps_V, backend: .mfa)
      var mfa_O = Tensor(copying: mps_O, backend: .mfa)
      var mfa_mask: Tensor<Real>?
      if let mps_mask {
        mfa_mask = Tensor(copying: mps_mask, backend: .mfa)
      }
      
      func act(
        _ Q: Tensor<Real>,
        _ K: Tensor<Real>,
        _ V: Tensor<Real>,
        _ O: inout Tensor<Real>,
        _ mask: Tensor<Real>?
      ) {
        var blockSparse = Self.useBlockSparsity
        blockSparse = blockSparse && (mask != nil)
        blockSparse = blockSparse && (TensorBackend.default == .mfa)
        
        O.attention(
          queries: Q, keys: K, values: V, mask: mask,
          transposeQ: Q_trans, transposeK: K_trans,
          transposeV: V_trans, transposeO: O_trans,
          blockSparse: blockSparse, accumulateInFloat: MFATestCase.accumulateInFloat)
      }
      
      if ghost {
        _ExecutionContext.withDefaultBackend(.mps) {
          TensorBackend.default.withGhostExecution {
            TensorBackend.default.markFirstCommand()
            act(mps_Q, mps_K, mps_V, &mps_O, mps_mask)
            TensorBackend.default.markLastCommand()
            _ = TensorBackend.default.synchronize()
          }
        }
        _ExecutionContext.withDefaultBackend(.mfa) {
          TensorBackend.default.withGhostExecution {
            TensorBackend.default.markFirstCommand()
            act(mfa_Q, mfa_K, mfa_V, &mfa_O, mfa_mask)
            TensorBackend.default.markLastCommand()
            _ = TensorBackend.default.synchronize()
          }
        }
      } else {
        _ExecutionContext.withDefaultBackend(.mps) {
          TensorBackend.default.markFirstCommand()
          act(mps_Q, mps_K, mps_V, &mps_O, mps_mask)
          TensorBackend.default.markLastCommand()
          _ = TensorBackend.default.synchronize()
        }
        _ExecutionContext.withDefaultBackend(.mfa) {
          TensorBackend.default.markFirstCommand()
          act(mfa_Q, mfa_K, mfa_V, &mfa_O, mfa_mask)
          TensorBackend.default.markLastCommand()
          _ = TensorBackend.default.synchronize()
        }
        
        let params = EuclideanDistanceParameters(
          averageMagnitude: 1.0,
          averageDeviation: 0.2,
          batchSize: H * (batchSize ?? 1))
        let failed = !mfa_O.isApproximatelyEqual(to: mps_O, parameters: params)
        if logProgress {
          var shapeRepr: String
          if let batchSize {
            shapeRepr = "\(batchSize)x\(R)x\(C)x\(H)x\(D)x\(DTypeRepr)"
          } else {
            shapeRepr = "\(R)x\(C)x\(H)x\(D)x\(DTypeRepr)"
          }
          var detailsRepr = transRepr
          switch randomMask {
          case .none:
            break
          case .upperTriangular:
            detailsRepr += ", triangular"
          case .blockSparse(let blockSize, _):
            let blockRepr = "\(blockSize)x\(blockSize)"
            detailsRepr += ", \(blockRepr) sparse"
          }
          let dist = mfa_O.euclideanDistance(to: mps_O)
          let distRepr = "- \(String(format: "%.3f", dist))"
          
          let passRer = failed ? "Failed test" : "Passed test"
          print("\(passRer): \(shapeRepr) (\(detailsRepr)) \(distRepr)")
          
          if failed {
            if mfa_O.hasNaN() {
              print(" - MFA has NaN")
            }
            if mps_O.hasNaN() {
              print(" - MPS has NaN")
            }
            
            switch randomMask {
            case .blockSparse(_, let sparsity):
              let percentRepr = String(format: "%.0f", sparsity * 100)
              print(" - Sparsity: \(percentRepr)%")
            default:
              break
            }
            
            if let mfa_mask {
              print(" - Mask dims: \(mfa_mask.shape)")
            }
          }
        }
        
        if !mfa_O.isApproximatelyEqual(to: mps_O, parameters: params) {
          if let batchSize {
            if !O_trans {
              print("Can't render this matrix because O not transposed.")
            } else {
              for batchIndex in 0..<batchSize {
                let _mfa_O = Tensor(
                  slicing: mfa_O,
                  indices: [batchIndex],
                  lastSlicedDim: 0,
                  backend: .mfa)
                let _mps_O = Tensor(
                  slicing: mps_O,
                  indices: [batchIndex],
                  lastSlicedDim: 0,
                  backend: .mps)
                
                for slice in 0..<H {
                  let prefix = "(\(batchIndex), \(slice))"
                  MPL_showComparison(
                    actual: _mfa_O, actualName: "\(prefix) MFA",
                    expected: _mps_O, expectedName: "\(prefix) MPS",
                    parameters: params, slice: PythonObject(slice),
                    transpose: O_trans)
                }
              }
            }
          } else {
            for slice in 0..<H {
              let prefix = "(\(slice))"
              MPL_showComparison(
                actual: mfa_O, actualName: "\(prefix) MFA",
                expected: mps_O, expectedName: "\(prefix) MPS",
                parameters: params, slice: PythonObject(slice),
                transpose: O_trans)
            }
//            fatalError("Tensors did not match.")
          }

        }
        
      }
    }
    
    for i in 0..<numTrials {
      testRandomSize(index: i, ghost: true)
    }
    for i in 0..<numTrials {
      testRandomSize(index: i, ghost: false)
    }
    
    let end = CACurrentMediaTime()
    let repr = String(format: "%.3f", end - start)
    print("Finished 'testRandomAttention' in \(repr) seconds.")
  }
  
//  let M = 100
//  let N = 50
//  let K = 25
//
//  typealias Real = Float32
//
//  let py_A = Tensor<Real>(shape: [K, M], randomUniform: 0..<1, backend: .numpy)
//  let py_B = Tensor<Real>(shape: [K, N], randomUniform: 0..<1, backend: .numpy)
//  var py_C = Tensor<Real>(zerosLike: [M, N], backend: .numpy)
//  _ExecutionContext.withDefaultBackend(.numpy) {
//    _ExecutionContext.profileCommands {
//      py_C.matmul(py_A, py_B, transposeA: true)
//    }
//  }
//
//  let mps_A = Tensor(copying: py_A, backend: .mps)
//  let mps_B = Tensor(copying: py_B, backend: .mps)
//  var mps_C = Tensor<Real>(zerosLike: [M, N], backend: .mps)
//  _ExecutionContext.withDefaultBackend(.mps) {
//    _ExecutionContext.profileCommands {
//      mps_C.matmul(mps_A, mps_B, transposeA: true)
//    }
//  }
//
//  let mfa_A = Tensor(copying: py_A, backend: .mfa)
//  let mfa_B = Tensor(copying: py_B, backend: .mfa)
//  var mfa_C = Tensor<Real>(zerosLike: [M, N], backend: .mfa)
//  _ExecutionContext.withDefaultBackend(.mfa) {
//    _ExecutionContext.profileCommands {
//      mfa_C.matmul(mfa_A, mfa_B, transposeA: true)
//    }
//  }
//
//  MPL_showBackends(
//    mfa: mfa_C, mps: mps_C, numpy: py_C,
//    parameters: .init(matrixK: K))
}
