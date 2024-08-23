//
//  Network.swift
//
//
//  Created by Philip Turner on 8/22/24.
//

import func Foundation.cos
import func Foundation.exp
import func Foundation.log
import func Foundation.sin

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
// idea 2: Φ = Σ_n Σ_d dO[n][d] * O[n][d]
//
// The first idea is the simplest. ∂Φ/∂O will be a rectangular matrix, where
// every element is 1. The second idea allows more rigorous validation tests.
// There could be a bug that doesn't show itself when ∂Φ/∂O has a pre-defined
// structure. The pointwise summation with a dO matrix would allow more
// variation in what values ∂Φ/∂O takes.
//
// idea 1: (∂Φ/∂O)[n][d] = 1
// idea 2: (∂Φ/∂O)[n][d] = dO[n][d]
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
  var rowDimension: Int?
  var columnDimension: Int?
  var headDimension: Int?
}

// Utility for testing the correctness of forward and backward attention.
// This has the same algorithmic complexity as standard attention (unlike
// finite differences), although it is not very parallel.
struct Network {
  let rowDimension: Int
  let columnDimension: Int
  let headDimension: Int
  
  var Q: [Float]
  var K: [Float]
  var V: [Float]
  var dO: [Float]
  
  init(descriptor: NetworkDescriptor) {
    guard let rowDimension = descriptor.rowDimension,
          let columnDimension = descriptor.columnDimension,
          let headDimension = descriptor.headDimension else {
      fatalError("Descriptor was incomplete.")
    }
    self.rowDimension = rowDimension
    self.columnDimension = columnDimension
    self.headDimension = headDimension
    
    // Randomly initialize Q, K, V, dO ∈ R^{NxD}
    Q = [Float](repeating: .zero, count: rowDimension * headDimension)
    K = [Float](repeating: .zero, count: columnDimension * headDimension)
    V = [Float](repeating: .zero, count: columnDimension * headDimension)
    dO = [Float](repeating: .zero, count: rowDimension * headDimension)
    
    for rowID in 0..<rowDimension {
      for d in 0..<headDimension {
        let matrixAddress = rowID * headDimension + d
        let randomNumberPair = Network.boxMullerTransform()
        Q[matrixAddress] = randomNumberPair[0]
        dO[matrixAddress] = randomNumberPair[1]
      }
    }
    
    for columnID in 0..<columnDimension {
      for d in 0..<headDimension {
        let matrixAddress = columnID * headDimension + d
        let randomNumberPair = Network.boxMullerTransform()
        K[matrixAddress] = randomNumberPair[0]
        V[matrixAddress] = randomNumberPair[1]
      }
    }
  }
  
  static func boxMullerTransform() -> SIMD2<Float> {
    let randomUniform = SIMD2<Float>.random(in: 0..<1)
    let logPart = log(randomUniform[0])
    let magnitudePart = (-2 * logPart).squareRoot()
    
    // This is an inefficient way to compute trigonometric quantities, but
    // we're not particularly focused on efficiency here.
    let anglePart = 2 * Float.pi * randomUniform[1]
    let cosPart = cos(anglePart)
    let sinPart = sin(anglePart)
    
    return SIMD2(
      magnitudePart * cosPart,
      magnitudePart * sinPart)
  }
}

// Utilities for materializing the attention matrix, one row at a time.
extension Network {
  func createMatrixSRow(rowID: Int) -> [Float] {
    var output = [Float](repeating: .zero, count: columnDimension)
    
    // Q * K^T
    for columnID in 0..<columnDimension {
      var dotProduct: Float = .zero
      for d in 0..<headDimension {
        let addressQ = rowID * headDimension + d
        let addressK = columnID * headDimension + d
        dotProduct += Q[addressQ] * K[addressK]
      }
      output[columnID] = dotProduct
    }
    
    return output
  }
  
  func createMatrixPRow(rowID: Int) -> [Float] {
    var output = createMatrixSRow(rowID: rowID)
    let scaleFactor = 1 / Float(headDimension).squareRoot()
    
    // Maximum
    var maximum: Float = -.greatestFiniteMagnitude
    for columnID in 0..<columnDimension {
      let value = scaleFactor * output[columnID]
      maximum = max(maximum, value)
    }
    
    // Sum
    var sum: Float = .zero
    for columnID in 0..<columnDimension {
      let value = scaleFactor * output[columnID]
      let expTerm = exp(value - maximum)
      sum += expTerm
    }
    
    // Log-Sum-Exp
    let lse = maximum + log(sum)
    for columnID in 0..<columnDimension {
      let value = scaleFactor * output[columnID]
      let expTerm = exp(value - lse)
      output[columnID] = expTerm
    }
    
    return output
  }
  
  func createLTerm(rowID: Int) -> Float {
    let matrixSRow = createMatrixSRow(rowID: rowID)
    let scaleFactor = 1 / Float(headDimension).squareRoot()
    
    // Maximum
    var maximum: Float = -.greatestFiniteMagnitude
    for columnID in 0..<columnDimension {
      let value = scaleFactor * matrixSRow[columnID]
      maximum = max(maximum, value)
    }
    
    // Sum
    var sum: Float = .zero
    for columnID in 0..<columnDimension {
      let value = scaleFactor * matrixSRow[columnID]
      let expTerm = exp(value - maximum)
      sum += expTerm
    }
    
    // Log-Sum-Exp
    let lse = maximum + log(sum)
    return lse
  }
  
  func createDerivativePRow(rowID: Int) -> [Float] {
    // dO * V^T
    var output = [Float](repeating: .zero, count: columnDimension)
    for columnID in 0..<columnDimension {
      var dotProduct: Float = .zero
      for d in 0..<headDimension {
        let addressO = rowID * headDimension + d
        let addressV = columnID * headDimension + d
        dotProduct += dO[addressO] * V[addressV]
      }
      output[columnID] = dotProduct
    }
    return output
  }
  
  func createDerivativeSRow(rowID: Int) -> [Float] {
    let matrixPRow = createMatrixPRow(rowID: rowID)
    
    // P * V
    var matrixORow = [Float](repeating: .zero, count: headDimension)
    for d in 0..<headDimension {
      var dotProduct: Float = .zero
      for columnID in 0..<columnDimension {
        let valueP = matrixPRow[columnID]
        let addressV = columnID * headDimension + d
        dotProduct += valueP * V[addressV]
      }
      matrixORow[d] = dotProduct
    }
    
    // dO * O
    var termD: Float = .zero
    for d in 0..<headDimension {
      let addressDerivativeO = rowID * headDimension + d
      termD += matrixORow[d] * dO[addressDerivativeO]
    }
    
    var derivativeSRow = [Float](repeating: .zero, count: columnDimension)
    let derivativePRow = createDerivativePRow(rowID: rowID)
    let scaleFactor = 1 / Float(headDimension).squareRoot()
    
    // P * (dP - D)
    for columnID in 0..<columnDimension {
      let valueP = matrixPRow[columnID]
      let valueDerivativeP = derivativePRow[columnID]
      var valueS = valueP * (valueDerivativeP - termD)
      
      valueS *= scaleFactor
      derivativeSRow[columnID] = valueS
    }
    
    return derivativeSRow
  }
  
  func createDTerm(rowID: Int) -> Float {
    let matrixPRow = createMatrixPRow(rowID: rowID)
    
    // P * V
    var matrixORow = [Float](repeating: .zero, count: headDimension)
    for d in 0..<headDimension {
      var dotProduct: Float = .zero
      for columnID in 0..<columnDimension {
        let valueP = matrixPRow[columnID]
        let addressV = columnID * headDimension + d
        dotProduct += valueP * V[addressV]
      }
      matrixORow[d] = dotProduct
    }
    
    // dO * O
    var termD: Float = .zero
    for d in 0..<headDimension {
      let addressDerivativeO = rowID * headDimension + d
      termD += matrixORow[d] * dO[addressDerivativeO]
    }
    return termD
  }
}

extension Network {
  // Compute O.
  func inferenceAttention() -> [Float] {
    var output = [Float](repeating: .zero, count: rowDimension * headDimension)
    for rowID in 0..<rowDimension {
      let matrixPRow = createMatrixPRow(rowID: rowID)
      
      // P * V
      var matrixORow = [Float](repeating: .zero, count: headDimension)
      for d in 0..<headDimension {
        var dotProduct: Float = .zero
        for columnID in 0..<columnDimension {
          let valueP = matrixPRow[columnID]
          let addressV = columnID * headDimension + d
          dotProduct += valueP * V[addressV]
        }
        matrixORow[d] = dotProduct
      }
      
      for d in 0..<headDimension {
        let valueO = matrixORow[d]
        let addressO = rowID * headDimension + d
        output[addressO] = valueO
      }
    }
    
    return output
  }
  
  // Compute Φ.
  func loss() -> Float {
    let O = inferenceAttention()
    var output: Float = .zero
    
    // Σ_n Σ_d dO[n][d] * O[n][d]
    for rowID in 0..<rowDimension {
      for d in 0..<headDimension {
        let address = rowID * headDimension + d
        output += dO[address] * O[address]
      }
    }
    return output
  }
  
  // Compute ∂Φ/∂V analytically.
  func derivativeV() -> [Float] {
    var output = [Float](
      repeating: .zero, count: columnDimension * headDimension)
    
    // P^T * dO
    for rowID in 0..<rowDimension {
      let matrixPRow = createMatrixPRow(rowID: rowID)
      
      for columnID in 0..<columnDimension {
        for d in 0..<headDimension {
          let addressV = columnID * headDimension + d
          let addressDerivativeO = rowID * headDimension + d
          
          var dotProduct = output[addressV]
          dotProduct += matrixPRow[columnID] * dO[addressDerivativeO]
          output[addressV] = dotProduct
        }
      }
    }
    return output
  }
  
  // Compute ∂Φ/∂K analytically.
  func derivativeK() -> [Float] {
    var output = [Float](
      repeating: .zero, count: columnDimension * headDimension)
    
    // dS^T * Q
    for rowID in 0..<rowDimension {
      let derivativeSRow = createDerivativeSRow(rowID: rowID)
      
      for columnID in 0..<columnDimension {
        for d in 0..<headDimension {
          let addressK = columnID * headDimension + d
          let addressQ = rowID * headDimension + d
          
          var dotProduct = output[addressK]
          dotProduct += derivativeSRow[columnID] * Q[addressQ]
          output[addressK] = dotProduct
        }
      }
    }
    return output
  }
  
  // Compute ∂Φ/∂Q analytically.
  func derivativeQ() -> [Float] {
    var output = [Float](
      repeating: .zero, count: rowDimension * headDimension)
    
    // dS * K
    for rowID in 0..<rowDimension {
      let derivativeSRow = createDerivativeSRow(rowID: rowID)
      
      var derivativeQRow = [Float](repeating: .zero, count: headDimension)
      for d in 0..<headDimension {
        var dotProduct: Float = .zero
        for columnID in 0..<columnDimension {
          let derivativeSValue = derivativeSRow[columnID]
          let addressK = columnID * headDimension + d
          dotProduct += derivativeSValue * K[addressK]
        }
        derivativeQRow[d] = dotProduct
      }
      
      for d in 0..<headDimension {
        let derivativeQValue = derivativeQRow[d]
        let addressQ = rowID * headDimension + d
        output[addressQ] = derivativeQValue
      }
    }
    
    return output
  }
}
