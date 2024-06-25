//
//  Network.swift
//  FlashAttention
//
//  Created by Philip Turner on 6/25/24.
//

import Numerics

// How do I even evaluate the gradient of a transformer?
//
// Attempt to find a simple model to compute and take the gradient of:
// https://arxiv.org/abs/2305.16380
//
// Simpler path: the "loss function" is a linear combination of the elements
// of the O matrix.
//
// Φ ∈ R^{1x1}
// O ∈ R^{NxD}
// idea 1: Φ = Σ_n Σ_d O[n][d]
// idea 2: Φ = Σ_n Σ_d C[n][d] * O[n][d]
//
// The first idea is the simplest. ∂Φ/∂O will be a rectangular matrix, where
// every element is 1. The second idea allows more rigorous validation tests.
// There could be a bug that doesn't show itself when ∂Φ/∂O has a pre-defined
// structure. The pointwise summation with a C matrix would allow more
// variation in what values ∂Φ/∂O takes.
//
// idea 1: (∂O/∂Φ)[n][d] = 1
// idea 2: (∂O/∂Φ)[n][d] = C[n][d]
//
// I now have the following:
// - Explicit functional form for the gradients.
// - Ability to calculate the derivative of every (intermediate) variable in
//   the attention mechanism, w.r.t. Φ
// - Numerical method for finding derivatives (finite differencing)
// - Analytical method for finding derivatives (backpropagation formula)
//
// I can set up a test, which compares the numerical and analytical gradients
// and confirms they are the same.
//
// Numerical method:
// - Peel back the neural network, removing the layers that generated Q/K/V.
//
// - Change one entry of Q, K, or V by +0.001.
// - Record the value for Φ.
// - Revert the change.
//
// - Change one entry of Q, K, or V by -0.001.
// - Record the value for Φ.
// - Revert the change.
//
// - Finite difference formula for gradient:
// - ∂Φ/∂X = ΔΦ/ΔX
// - ΔΦ/ΔX = (Φ(+0.001) - Φ(-0.001)) / (X + 0.001 - (X - 0.001))
// - ΔΦ/ΔX = (Φ(+0.001) - Φ(-0.001)) / 0.002

struct NetworkDescriptor {
  var N: Int?
  var D: Int?
}

// Utility for testing the correctness of forward and backward attention.
// This has the same algorithmic complexity as standard attention (unlike
// finite differences), although it is not very parallel.
struct Network {
  let N: Int
  let D: Int
  
  var Q: [Float]
  var K: [Float]
  var V: [Float]
  var C: [Float]
  
  init(descriptor: NetworkDescriptor) {
    guard let N = descriptor.N,
          let D = descriptor.D else {
      fatalError("Descriptor was incomplete.")
    }
    self.N = N
    self.D = D
    
    // Randomly initialize Q, K, V, C ∈ R^{NxD}
    Q = [Float](repeating: .zero, count: N * D)
    K = [Float](repeating: .zero, count: N * D)
    V = [Float](repeating: .zero, count: N * D)
    C = [Float](repeating: .zero, count: N * D)
    
    for n in 0..<N {
      for d in 0..<D {
        let matrixAddress = n * D + d
        let randomQK = Network.boxMullerTransform()
        let randomVC = Network.boxMullerTransform()
        Q[matrixAddress] = randomQK[0]
        K[matrixAddress] = randomQK[1]
        V[matrixAddress] = randomVC[0]
        C[matrixAddress] = randomVC[1]
      }
    }
  }
  
  static func boxMullerTransform() -> SIMD2<Float> {
    let randomUniform = SIMD2<Float>.random(in: 0..<1)
    let logPart = Float.log(randomUniform[0])
    let magnitudePart = (-2 * logPart).squareRoot()
    
    // This is an inefficient way to compute trigonometric quantities, but
    // we're not particularly focused on efficiency here.
    let anglePart = 2 * Float.pi * randomUniform[1]
    let cosPart = Float.cos(anglePart)
    let sinPart = Float.sin(anglePart)
    
    return SIMD2(
      magnitudePart * cosPart,
      magnitudePart * sinPart)
  }
}

// Utilities for materializing the attention matrix, one row at a time.
extension Network {
  func createAttentionMatrixRow(rowID: Int) -> [Float] {
    var output = [Float](repeating: .zero, count: N)
    let scaleFactor = 1 / Float(D).squareRoot()
    
    // Q * K
    for columnID in 0..<N {
      var dotProduct: Float = .zero
      for d in 0..<D {
        let addressQ = rowID * D + d
        let addressK = columnID * D + d
        dotProduct += Q[addressQ] * K[addressK]
      }
      dotProduct *= scaleFactor
      output[columnID] = dotProduct
    }
    
    // softmax
    do {
      var maximum: Float = -.greatestFiniteMagnitude
      for columnID in 0..<N {
        let value = output[columnID]
        maximum = max(maximum, value)
      }
      
      var sum: Float = .zero
      for columnID in 0..<N {
        let value = output[columnID]
        let expTerm = Float.exp(value - maximum)
        sum += expTerm
      }
      
      for columnID in 0..<N {
        let value = output[columnID]
        let expTerm = Float.exp(value - maximum)
        output[columnID] = expTerm / sum
      }
    }
    
    return output
  }
  
  func createDerivativeSRow(rowID: Int) -> [Float] {
    let attentionMatrixRow = createAttentionMatrixRow(rowID: rowID)
    
    // P * V
    var outputMatrixRow = [Float](repeating: .zero, count: D)
    for d in 0..<D {
      var dotProduct: Float = .zero
      for columnID in 0..<N {
        let attentionMatrixValue = attentionMatrixRow[columnID]
        let addressV = columnID * D + d
        dotProduct += attentionMatrixValue * V[addressV]
      }
      outputMatrixRow[d] = dotProduct
    }
    
    // dO = C
    var derivativeORow = [Float](repeating: .zero, count: D)
    for d in 0..<D {
      let addressC = rowID * D + d
      derivativeORow[d] = C[addressC]
    }
    
    // D = dO^T O
    var termD: Float = .zero
    for d in 0..<D {
      termD += outputMatrixRow[d] * derivativeORow[d]
    }
    
    // dP = dO V^T
    var derivativePRow = [Float](repeating: .zero, count: N)
    for columnID in 0..<N {
      var dotProduct: Float = .zero
      for d in 0..<D {
        let addressV = columnID * D + d
        dotProduct += derivativeORow[d] * V[addressV]
      }
      derivativePRow[columnID] = dotProduct
    }
    
    let scaleFactor = 1 / Float(D).squareRoot()
    
    // dS = P * (dP - D)
    var derivativeSRow = [Float](repeating: .zero, count: N)
    for n in 0..<N {
      let valueP = attentionMatrixRow[n]
      let valueDerivativeP = derivativePRow[n]
      var valueS = valueP * (valueDerivativeP - termD)
      
      valueS *= scaleFactor
      derivativeSRow[n] = valueS
    }
    
    return derivativeSRow
  }
}

extension Network {
  // Performs self-attention with the current values of Q, K, and V.
  // - S = QK^T
  // - P = softmax(S)
  // - O = PV
  //
  // Returns O, a matrix with dimensions N * D.
  func inferenceAttention() -> [Float] {
    var output = [Float](repeating: .zero, count: N * D)
    for rowID in 0..<N {
      let attentionMatrixRow = createAttentionMatrixRow(rowID: rowID)
      
      // P * V
      var outputMatrixRow = [Float](repeating: .zero, count: D)
      for d in 0..<D {
        var dotProduct: Float = .zero
        for columnID in 0..<N {
          let attentionMatrixValue = attentionMatrixRow[columnID]
          let addressV = columnID * D + d
          dotProduct += attentionMatrixValue * V[addressV]
        }
        outputMatrixRow[d] = dotProduct
      }
      
      for d in 0..<D {
        let outputMatrixValue = outputMatrixRow[d]
        let addressO = rowID * D + d
        output[addressO] = outputMatrixValue
      }
    }
    
    return output
  }
  
  // Computes the loss from the stored value of C, and the entered value of O.
  // - Φ = Σ_n Σ_d C[n][d] * O[n][d]
  func loss(O: [Float]) -> Float {
    var output: Float = .zero
    for n in 0..<N {
      for d in 0..<D {
        let address = n * D + d
        output += C[address] * O[address]
      }
    }
    return output
  }
  
  // dΦ/dV = P^T dΦ/dO
  func derivativeV() -> [Float] {
    var output = [Float](repeating: .zero, count: N * D)
    for columnID in 0..<N {
      let attentionMatrixRow = createAttentionMatrixRow(rowID: columnID)
      
      for n in 0..<N {
        for d in 0..<D {
          let addressV = n * D + d
          let addressC = columnID * D + d
          
          var dotProduct = output[addressV]
          dotProduct += attentionMatrixRow[n] * C[addressC]
          output[addressV] = dotProduct
        }
      }
    }
    return output
  }
  
  // dΦ/dK = dS^T Q
  func derivativeK() -> [Float] {
    var output = [Float](repeating: .zero, count: N * D)
    for columnID in 0..<N {
      let derivativeSRow = createDerivativeSRow(rowID: columnID)
      
      for n in 0..<N {
        for d in 0..<D {
          let addressK = n * D + d
          let addressQ = columnID * D + d
          
          var dotProduct = output[addressK]
          dotProduct += derivativeSRow[n] * Q[addressQ]
          output[addressK] = dotProduct
        }
      }
    }
    return output
  }
  
  // dΦ/dQ = dS K
  func derivativeQ() -> [Float] {
    var output = [Float](repeating: .zero, count: N * D)
    for rowID in 0..<N {
      let derivativeSRow = createDerivativeSRow(rowID: rowID)
      
      // dS * K
      var derivativeQRow = [Float](repeating: .zero, count: D)
      for d in 0..<D {
        var dotProduct: Float = .zero
        for columnID in 0..<N {
          let derivativeSValue = derivativeSRow[columnID]
          let addressK = columnID * D + d
          dotProduct += derivativeSValue * K[addressK]
        }
        derivativeQRow[d] = dotProduct
      }
      
      for d in 0..<D {
        let derivativeQValue = derivativeQRow[d]
        let addressQ = rowID * D + d
        output[addressQ] = derivativeQValue
      }
    }
    
    return output
  }
}
