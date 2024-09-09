//
//  ContentView.swift
//  On-Device-NLP
//
//  Created by 이규환 on 6/25/24.
//

import SwiftUI

struct ContentView: View {
    
    var body: some View {
        VStack {
          Button(action: generate) {
            Text("Click").font(.title)
          }.buttonStyle(.borderedProminent)
        }
        .padding()
    }
  
  func generate() {
      do {
          let soraPipeline = try SoraPipeline(resourcesAt: Bundle.main.bundleURL)
          print("Click")
          soraPipeline.sample(prompt: "Please Test T5...")
      } catch {
          print("Error: Can't initiallize SoraPipeline")
      }
    }
}

#Preview {
    ContentView()
}
