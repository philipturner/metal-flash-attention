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
  
  var networkDesc = NetworkDescriptor()
  networkDesc.N = N
  networkDesc.D = D
  var network = Network(descriptor: networkDesc)
  
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
  print("attention matrix:")
  for rowID in 0..<N {
    let attentionMatrixRow = network.createAttentionMatrixRow(rowID: rowID)
    for n in 0..<N {
      var repr = String(format: "%.3f", attentionMatrixRow[n])
      while repr.count < 8 {
        repr = " " + repr
      }
      print(repr, terminator: " ")
    }
    print()
  }
  
  print()
  print("dS:")
  for n in 0..<N {
    network.createDerivativeSRow(rowID: n)
  }
  
  print()
  print("V^T")
  printMatrix(network.V)
  
  print()
  print("C^T")
  printMatrix(network.C)
  
  do {
    print()
    print("O")
    let O = network.inferenceAttention()
    printMatrix(O)
    
    let Φ = network.loss(O: O)
    print()
    print("Φ:", Φ)
  }
  
  do {
    let derivativeV = network.derivativeV()
    
    print()
    for elementID in 0..<(N * D) {
      // Test the correctness of the derivatives component-by-component.
      let savedValue = network.V[elementID]
      
      // When comparing against the analytical formula, show the numerical
      // derivatives for each step size. There doesn't appear to be a
      // specific size that works for every case.
      let stepSizes: [Float] = [1, 0.1, 0.01, 0.001]
      var derivatives: [Float] = []
      for stepSize in stepSizes {
        network.V[elementID] = savedValue + stepSize
        let O1 = network.inferenceAttention()
        let Φ1 = network.loss(O: O1)
        
        network.V[elementID] = savedValue - stepSize
        let O2 = network.inferenceAttention()
        let Φ2 = network.loss(O: O2)
        
        let derivative = (Φ1 - Φ2) / (stepSize - (-stepSize))
        derivatives.append(derivative)
      }
      network.V[elementID] = savedValue
      
      var elementIDRepr = "\(elementID)"
      while elementIDRepr.count < 5 {
        elementIDRepr = " " + elementIDRepr
      }
      
      print("dΦ/dV[\(elementIDRepr)]:", terminator: " ")
      for derivative in derivatives {
        var repr = String(format: "%.3f", derivative)
        while repr.count < 8 {
          repr = " " + repr
        }
        print(repr, terminator: " ")
      }
      
      do {
        var repr = String(format: "%.3f", derivativeV[elementID])
        while repr.count < 8 {
          repr = " " + repr
        }
        print("| ", repr)
      }
    }
  }
}

struct NetworkDescriptor {
  var N: Int?
  var D: Int?
}

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

extension Network {
  func createAttentionMatrixRow(rowID: Int) -> [Float] {
    var output = [Float](repeating: .zero, count: N)
    
    // Q * K
    for columnID in 0..<N {
      var dotProduct: Float = .zero
      for d in 0..<D {
        let addressQ = rowID * D + d
        let addressK = columnID * D + d
        dotProduct += Q[addressQ] * K[addressK]
      }
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
    
    // dO
    var derivativeORow = [Float](repeating: .zero, count: D)
    for d in 0..<D {
      let addressC = rowID * D + d
      derivativeORow[d] = C[addressC]
    }
    
    // dO^T O
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
      
      var repr = String(format: "%.3f", derivativePRow[columnID])
      while repr.count < 8 {
        repr = " " + repr
      }
      print(repr, terminator: " ")
    }
    print()
    
    return []
  }
}
