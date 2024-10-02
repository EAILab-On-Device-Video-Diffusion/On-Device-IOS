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

/*
That is the code for the video conversion test.

struct ContentView: View {
    @StateObject private var tensor2vidConverter = Tensor2Vid()
    
    var body: some View {
        VStack {
            if let videoURL = tensor2vidConverter.videoURL {
                VideoPlayerView(url: videoURL)
            } else {
                Text("video is not generated yet")
            }
            
            Button("generate video") {
                Task {
                
                // MLMultiArray(shape: [1, 3, 256, 256], dataType: .float32)
                    let multiArray = tensor2vidConverter.generateRandomInput(frameCount: 1, width: 256, height: 256)
                    await tensor2vidConverter.convertToVideo(multiArray: multiArray!)
                }
            }
        }
        .padding()
    }
}



*/