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
  
  // Forward attention layer:
  // - Randomly initialize Q, K, V ∈ R^{NxD}
  //   - Batch the random initialization of C with Q/K/V. We will use C in the
  //     next layer.
  // - S = QK^T
  // - P = softmax(S)
  // - O = PV
  
  // TODO: Create single-core, scalar CPU code that randomly initializes the
  // input matrices. Then, it multiplies the relevant matrices and performs a
  // softmax operation. To enable finite differencing, the code should work in
  // FP32 (no 16-bit types).
  
  let N: Int = 128
  let D: Int = 16
  
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
  
  // Generate a histogram from all the values in Q/K/V/C. Confirm that they
  // match the bell curve.
  do {
    var histogram = [Int](repeating: .zero, count: 200)
    let population = Q + K + V + C
    for sample in population {
      var output = sample
      
      // Shift the average from x = 0 to x = 10.
      output += 10
      
      // Scale from 8 < x < 12 to 80 < x < 120.
      output *= 10
      
      // Round toward negative infinity.
      var slotID = Int(output.rounded(.down))
      
      // Clamp to within array bounds.
      slotID = max(0, min(199, slotID))
      
      histogram[slotID] += 1
    }
    
    var cumulativeSum: Int = .zero
    for slotID in histogram.indices {
      var slotRepr = "\(slotID)"
      while slotRepr.count < 3 {
        slotRepr = " " + slotRepr
      }
      print("histogram[\(slotRepr)]", terminator: " | ")
      
      let lowerBound = Float(slotID) / 10 - 10
      let upperBound = lowerBound + 0.1
      var upperBoundRepr = String(format: "%.2f", upperBound)
      while upperBoundRepr.count < 5 {
        upperBoundRepr = " " + upperBoundRepr
      }
      print("x = \(upperBoundRepr)", terminator: " | ")
      
      let histogramValue = histogram[slotID]
      cumulativeSum += histogramValue
      let proportion = Float(cumulativeSum) / Float(population.count)
      let percentage = 100 * proportion
      var percentageRepr = String(format: "%.1f", percentage)
      while percentageRepr.count < 5 {
        percentageRepr = " " + percentageRepr
      }
      print(percentageRepr + "%")
    }
  }
}
