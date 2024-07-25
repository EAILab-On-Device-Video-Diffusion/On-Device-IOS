//
//  ContentView.swift
//  On-Device-NLP
//
//  Created by 이규환 on 6/25/24.
//

import SwiftUI

struct ContentView: View {
    @StateObject var stditGenerator = StditGenerator()
    var body: some View {
        VStack {
          Button(action: generate) {
            Text("Click").font(.title)
          }.buttonStyle(.borderedProminent)
        }
        .padding()
    }
  
  func generate() {
      print("Click")
      stditGenerator.modelProcessing()
    }
}

#Preview {
    ContentView()
}
