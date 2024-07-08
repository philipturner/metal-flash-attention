//
//  FiniteDifferencingTest.swift
//  FlashAttention
//
//  Created by Philip Turner on 6/25/24.
//

// Tests the correctness of analytical derivatives, by comparing them to
// numerical derivatives.

#if false
func executeScript() {
  // Define the problem dimensions.
  let N: Int = 64
  let D: Int = 8
  
  var networkDesc = NetworkDescriptor()
  networkDesc.N = N
  networkDesc.D = D
  var network = Network(descriptor: networkDesc)
  
  // Displays a matrix with dimensions N * D.
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
    let matrixPRow = network.createMatrixPRow(rowID: rowID)
    for n in 0..<N {
      var repr = String(format: "%.3f", matrixPRow[n])
      while repr.count < 8 {
        repr = " " + repr
      }
      print(repr, terminator: " ")
    }
    print()
  }
  
  print()
  print("dS:")
  for rowID in 0..<N {
    let derivativeSRow = network.createDerivativeSRow(rowID: rowID)
    for n in 0..<N {
      var repr = String(format: "%.3f", derivativeSRow[n])
      while repr.count < 8 {
        repr = " " + repr
      }
      print(repr, terminator: " ")
    }
    print()
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
    let derivativeQ = network.derivativeQ()
    
    print()
    for elementID in 0..<(N * D) {
      // Test the correctness of the derivatives component-by-component.
      let savedValue = network.Q[elementID]
      
      // When comparing against the analytical formula, show the numerical
      // derivatives for each step size. There doesn't appear to be a
      // specific size that works for every case.
      let stepSizes: [Float] = [1, 0.1, 0.01, 0.001]
      var derivatives: [Float] = []
      for stepSize in stepSizes {
        network.Q[elementID] = savedValue + stepSize
        let O1 = network.inferenceAttention()
        let Φ1 = network.loss(O: O1)
        
        network.Q[elementID] = savedValue - stepSize
        let O2 = network.inferenceAttention()
        let Φ2 = network.loss(O: O2)
        
        let derivative = (Φ1 - Φ2) / (stepSize - (-stepSize))
        derivatives.append(derivative)
      }
      network.Q[elementID] = savedValue
      
      var elementIDRepr = "\(elementID)"
      while elementIDRepr.count < 5 {
        elementIDRepr = " " + elementIDRepr
      }
      
      print("dΦ/dQ[\(elementIDRepr)]:", terminator: " ")
      for derivative in derivatives {
        var repr = String(format: "%.3f", derivative)
        while repr.count < 8 {
          repr = " " + repr
        }
        print(repr, terminator: " ")
      }
      
      do {
        var repr = String(format: "%.3f", derivativeQ[elementID])
        while repr.count < 8 {
          repr = " " + repr
        }
        print("| ", repr)
      }
    }
  }
}
#endif
