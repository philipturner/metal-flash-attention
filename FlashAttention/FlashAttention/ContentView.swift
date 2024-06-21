//
//  ContentView.swift
//  FlashAttention
//
//  Created by Philip Turner on 6/20/24.
//

import SwiftUI

struct ContentView: View {
  var body: some View {
    VStack {
      Image(systemName: "globe")
        .imageScale(.large)
        .foregroundStyle(.tint)
      Text(ContentView.createText())
    }
    .padding()
  }
  
  /// Hijack SwiftUI, so the application exits before rendering anything to
  /// the screen. This is a command-line application. It just executes within
  /// the SwiftUI run loop to be deployable on iOS.
  private static func createText() -> String {
    executeScript()
    exit(0)
  }
}

