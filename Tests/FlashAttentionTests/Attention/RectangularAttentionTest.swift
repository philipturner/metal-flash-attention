import XCTest
import FlashAttention

final class RectangularAttentionTest: XCTestCase {
  // Tests random permutations of transpose state and input/output sequence
  // length. Just like the old MFA test suite.
  //
  // For simplicity, we are only testing FP32. This removes the need to worry
  // about numerical rounding error. With mixed precision, the rounding
  // error scales with problem dimension in predictable ways. We have
  // discovered predictive formulae for GEMM through trial and error.
  
}

/// Run a test with the specified configuration.
private func runCorrectnessTest(descriptor: AttentionDescriptor) {
  // Check that all properties of the descriptor have been set.
  guard let matrixDimensions = descriptor.matrixDimensions,
        let transposeState = descriptor.transposeState else {
    fatalError("Descriptor was incomplete.")
  }
  guard !descriptor.lowPrecisionInputs,
        !descriptor.lowPrecisionIntermediates else {
    fatalError("Mixed precision is not supported.")
  }
  
  
}
