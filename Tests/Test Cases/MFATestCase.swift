//
//  MFATestCase.swift
//  MetalFlashAttention
//
//  Created by Philip Turner on 6/27/23.
//

import Foundation

class MFATestCase {
  // Global setting for the precision used in tests.
  #if arch(arm64)
  typealias Real = Float32
  #else
  typealias Real = Float
  #endif
  
  class func typeDescription() -> String {
    fatalError("Not implemented.")
  }
  
  enum TestSpeed {
    // Only run quick smoke tests.
    case quick
    
    // Run quick performance tests.
    case long
    
    // Run long performance tests.
    case veryLong
  }
  
  static func runTests(speed: TestSpeed) {
    let testCases: [MFATestCase] = [
      CorrectnessTests(),
//      AttentionPerfTests(),
//      GEMMPerfTests(),
    ]
    
    for testCase in testCases {
      testCase.runQuickTests()
      if speed == .quick { continue }
      testCase.runLongTests()
      if speed == .long { continue }
      testCase.runVeryLongTests()
    }
  }
  
  func runQuickTests() {
    
  }
  
  func runLongTests() {
    
  }
  
  func runVeryLongTests() {
    
  }
  
  func pass(_ function: StaticString = #function) {
    // do nothing
  }
  
  func fail(_ function: StaticString = #function) {
    var message = "Test case '\(Self.typeDescription())', "
    message += "function '\(function)' failed."
    print(message)
  }
}
