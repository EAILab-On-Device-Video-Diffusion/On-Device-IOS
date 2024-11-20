//
//  ContentView.swift
//  On-Device-NLP
//
//  Created by 이규환 on 6/25/24.
//

import SwiftUI
import Foundation

struct ContentView: View {
    var prompts: [String] = []
    var category: String = "animal"
    var logdir: String = "/Users/ijeong/workspace/on-device-diffusion/samples/"
    @State var prompt: String = "a serene underwater scene featuring a sea turtle swimming through a coral reef. the turtle, with its greenish-brown shell aesthetic score: 6.5."
    @StateObject private var tensor2vidConverter = Tensor2Vid()

    init() {
        // change log dir
        logdir = "\(logdir)/\(category)"
        
        // load prompts
        let promptURL = Bundle.main.bundleURL.appending(path: "animal.txt")
        if !FileManager.default.fileExists(atPath: promptURL.path) {
            print("File does not exist at path: \(promptURL.path)")
            return
        }
        do {
            print("Try to read \(promptURL.path)")
            let content = try String(contentsOfFile: promptURL.path, encoding: .ascii)
            print(content)
            
            // Split content into lines
            // prompts.append() content.components(separatedBy: .newlines)
            
            let lines = content.components(separatedBy: .newlines)
            prompts.append(contentsOf: lines)
            print(prompts)
       
        } catch {
            print("Error loading file: \(error)")
        }
    }

    var body: some View {
 
        VStack {
            // GUI test
            //  if let videoURL = tensor2vidConverter.videoURL {
            //      VideoPlayerView(url: videoURL)
            //  } else {
            //      TextField("Enter prompt,but default exists", text: $prompt)
            //      .padding()
            //      .background(Color(uiColor: .secondarySystemBackground))
            //  }
             // Button(action: generate) {
             //     Text("Click").font(.title)
             // }.buttonStyle(.borderedProminent)

            // for experiments
            Text("Lines from animal.txt:")
                            .font(.headline)
                            .padding()
            
            List(prompts, id: \.self) { prompt in
                Text(prompt)
            }
            .onAppear {
                generate()
            }

        }
        .padding()
    }
  
  func generate() {
        do {
            let soraPipeline = try SoraPipeline(resourcesAt: Bundle.main.bundleURL, videoConverter: tensor2vidConverter)
            // print("Click")
            
//            for (index, prompt) in prompts.enumerated() {
//                print("prompt: \(prompt)")
//                var filename: String = "sample-\(index).mp4"
//                // soraPipeline.sample(prompt: prompt, logdir: logdir, filename: filename)
//                soraPipeline.sample(prompt: prompt, logdir: logdir)
//                print("sample done")
//            }
            soraPipeline.sample(prompts: prompts, logdir: logdir)
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
