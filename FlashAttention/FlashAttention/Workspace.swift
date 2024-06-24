//
//  Workspace.swift
//  FlashAttention
//
//  Created by Philip Turner on 6/20/24.
//

import Metal
import Numerics
import QuartzCore

/// The repo author's own workspace for running tests and developing kernels.
/// The contents of this function have no meaning, and ideally will be blank
/// when the 'main' branch is in a stable state. Clients can utilize this
/// function to script tests in their fork.
func executeScript() {
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
  // There could be a bug that doesn't show itself when ∂Φ/∂O has a specific
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
   
  // Define the problem dimensions.
  let N: Int = 10
  let D: Int = 3
  
  // Randomly initialize Q, K, V, C ∈ R^{NxD}
  var Q = [Float](repeating: .zero, count: N * D)
  var K = [Float](repeating: .zero, count: N * D)
  var V = [Float](repeating: .zero, count: N * D)
  var C = [Float](repeating: .zero, count: N * D)
  
  func boxMullerTransform() -> SIMD2<Float> {
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
  
  for n in 0..<N {
    for d in 0..<D {
      let matrixAddress = n * D + d
      let randomQK = boxMullerTransform()
      let randomVC = boxMullerTransform()
      Q[matrixAddress] = randomQK[0]
      K[matrixAddress] = randomQK[1]
      V[matrixAddress] = randomVC[0]
      C[matrixAddress] = randomVC[1]
    }
  }
  
  // Forward attention layer:
  // - S = QK^T
  // - P = softmax(S)
  // - O = PV
  var O = [Float](repeating: .zero, count: N * D)
  
  print()
  print("attention matrix:")
  for rowID in 0..<N {
    var attentionMatrixRow = [Float](repeating: .zero, count: N)
    
    // Q * K
    for columnID in 0..<N {
      var dotProduct: Float = .zero
      for d in 0..<D {
        let addressQ = rowID * D + d
        let addressK = columnID * D + d
        dotProduct += Q[addressQ] * K[addressK]
      }
      attentionMatrixRow[columnID] = dotProduct
    }
    
    // softmax
    do {
      var maximum: Float = -.greatestFiniteMagnitude
      for columnID in 0..<N {
        let value = attentionMatrixRow[columnID]
        maximum = max(maximum, value)
      }
      
      var sum: Float = .zero
      for columnID in 0..<N {
        let value = attentionMatrixRow[columnID]
        let expTerm = Float.exp(value - maximum)
        sum += expTerm
      }
      
      for columnID in 0..<N {
        let value = attentionMatrixRow[columnID]
        let expTerm = Float.exp(value - maximum)
        attentionMatrixRow[columnID] = expTerm / sum
        
        // Display the attention matrix.
        let matrixValue = attentionMatrixRow[columnID]
        var repr = String(format: "%.3f", matrixValue)
        while repr.count < 8 {
          repr = " " + repr
        }
        print(repr, terminator: " ")
      }
      print()
    }
    
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
      O[addressO] = outputMatrixValue
    }
  }
  
  // Displays a matrix with dimensions N * d.
  func printMatrix(_ matrix: [Float]) {
    for d in 0..<D {
      for n in 0..<N {
        let matrixAddress = n * D + d
        let matrixValue = matrix[matrixAddress]
        var repr = String(format: "%.3f", matrixValue)
        while repr.count < 8 {
          repr = " " + repr
        }
        print(repr, terminator: " ")
      }
      print()
    }
  }
  
  print()
  print("Q^T:")
  printMatrix(Q)
  
  print()
  print("K^T:")
  printMatrix(K)
  
  print()
  print("V^T:")
  printMatrix(V)
  
  print()
  print("O^T:")
  printMatrix(O)
  
  print()
  print("C^T:")
  printMatrix(C)
  
  // Loss function layer:
  // - Φ = Σ_n Σ_d C[n][d] * O[n][d]
  var Φ: Float = .zero
  for n in 0..<1 {
    for d in 0..<D {
      let address = n * D + d
      Φ += O[address] * C[address]
    }
  }
  print()
  print("Φ:", Φ)
  
  // TODO: Wrap the above code in a function, which accepts Q/K/V/O as
  // (descriptor) arguments. Use it as a primitive for finite differencing.
  // Show the limit of the derivative for each variable as h approaches 0.
}
