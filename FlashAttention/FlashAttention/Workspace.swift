//
//  Workspace.swift
//  FlashAttention
//
//  Created by Philip Turner on 6/20/24.
//

import Metal
import QuartzCore

/// The repo author's own workspace for running tests and developing kernels.
/// The contents of this function have no meaning, and ideally will be blank
/// when the 'main' branch is in a stable state. Clients can utilize this
/// function to script tests in their fork.
func executeScript() {
  // TODO: Preserve this commentary in the source file where the test is
  // archived/maintained.
  //
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
  // The first idea is the simplest. ∂O/∂Φ will be a rectangular matrix, where
  // every element is 1. The second idea allows more rigorous validation tests.
  // There could be a bug that doesn't show itself when ∂O/∂Φ has a specific
  // structure. The pointwise summation with a C matrix would allow more
  // variation in what values ∂O/∂Φ takes.
  //
  // idea 1: (∂O/∂Φ)[n][d] = 1
  // idea 2: (∂O/∂Φ)[n][d] = 1 / C[n][d]
  //
  // I now have the following:
  // - Explicit functional form for the gradients.
  // - Ability to calculate the derivative of every (intermediate) variable in
  //   the attention mechanism, w.r.t. Φ
  // - Numerical method for finding derivatives (finite differencing)
  // - Analytical method for finding derivatives (backpropagation formula)
  //
  // I can set up a test, which compares the numerical and analytical gradients
  // and confirms they are the same. If the output doesn't depend on a certain
  // variable very much, the finite difference might end up dividing by zero.
  // Therefore, the reciprocal of ∂X/∂Φ may be important for analysis.
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
  // - ∂X/∂Φ = ΔX/ΔΦ
  // - ΔX/ΔΦ = (X + 0.001 - (X - 0.001)) / (Φ(+0.001) - Φ(-0.001))
  // - ΔX/ΔΦ = 0.002 / (Φ(+0.001) - Φ(-0.001))
  //
  // I know from solving SCF problems, that you minimize the energy
  // functional by finding the derivative ∂E/∂X. Here, we're computing the
  // reciprocal for some reason (∂X/∂E where E = Φ). I would assume that
  // when updating MLP weights, you invert the value ∂X/∂Φ. You now have ∂Φ/∂X,
  // the rate of change in the loss function with respect to the variational
  // parameter. If this rate is very high, you change the variable very much.
}
