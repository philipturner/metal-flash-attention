//
//  PerformanceTests.swift
//  MetalFlashAttention
//
//  Created by Philip Turner on 6/27/23.
//

import Metal

class PerformanceTests: MFATestCase {
  override class func typeDescription() -> String {
    "PerformanceTests"
  }
  
  override func runVeryLongTests() {
    testHGEMMSpeed()
  }
  
  func testHGEMMSpeed() {
    let flops = MetalContext.global.infoDevice.flops
    let referenceThreshold: Float = 7000e9
    let performanceRatio = Float(flops) / 10616832000000.0
    let performanceThreshold = performanceRatio * referenceThreshold
    print("Performance threshold: \(performanceThreshold)")
  }
}
