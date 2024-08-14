//
//  SoraPipeline.swift
//  On-Device-OpenSora
//
//  Created by hanbitchan on 8/14/24.
//

import Foundation
import CoreML
import Accelerate
import Tokenizers
import Hub

struct ResourceURLs {
    public let textEncoderT5URL: URL
    public let stditURL: URL
    public let decoderURL: URL
    public let configT5URL: URL
    public let dataT5URL: URL
    
    // To do: change the file format to mlpackage
    public init(resourcesAt baseURL: URL) {
        textEncoderT5URL = baseURL.appending(path: "TextEncoderT5.mlmodelc")
        stditURL = baseURL.appending(path: "MultiModalDiffusionTransformer.mlmodelc")
        decoderURL = baseURL.appending(path: "VAEDecoder.mlmodelc")
        configT5URL = baseURL.appending(path: "tokenizer_config.json")
        dataT5URL = baseURL.appending(path: "tokenizer.json")
    }
}

public struct SoraPipeline {
  var textEncoderT5: TextEncoderT5?
  
  init(
          resourcesAt baseURL: URL,
          configuration config: MLModelConfiguration = .init(),
          reduceMemory: Bool = false
  ) throws {
    let urls = ResourceURLs(resourcesAt: baseURL)

    if FileManager.default.fileExists(atPath: urls.configT5URL.path),
       FileManager.default.fileExists(atPath: urls.dataT5URL.path),
       FileManager.default.fileExists(atPath: urls.textEncoderT5URL.path)
    {
      let tokenizerT5 = try PreTrainedTokenizer(tokenizerConfig: Config(fileURL: urls.configT5URL), tokenizerData: Config(fileURL: urls.dataT5URL))
      textEncoderT5 = TextEncoderT5(tokenizer: tokenizerT5, modelURL: urls.textEncoderT5URL, config: config)
    } else {
      textEncoderT5 = nil
    }
    
    // To do: STDit Model
    let STDit = STDit(modelURL: urls.stditURL, config: config)
    
    // To do: VAE for decoding video
    let VAEDecoder = VAEDecoder(modelURL: urls.decoderURL, config: config)
    
    // To do: generate Video
    
  }
}
