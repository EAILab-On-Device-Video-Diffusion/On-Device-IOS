//
//  ContentView.swift
//  OnDevice-IOS
//
//  Created by 이규환 on 4/24/24.
//

import SwiftUI

struct ContentView: View {
    @StateObject var imageGenerator = ImageGenerator()

    var body: some View {
        TabView {
            TextToImageView(imageGenerator: imageGenerator)
                .tabItem {
                    Image(systemName: "text.below.photo.fill")
                    Text("Text to Image")
                }
            ImageToImageView(imageGenerator: imageGenerator)
                .tabItem {
                    Image(systemName: "photo.stack.fill")
                    Text("Image to Image")
                }
        }
        .accentColor(.purple)
    }
}


#Preview {
    ContentView()
}
