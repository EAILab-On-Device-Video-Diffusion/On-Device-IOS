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
    public let stditURL: URL
    public let decoderURL: URL
    public let configT5URL: URL
    public let dataT5URL: URL
    public let embedURL: URL
    public let finalNormURL: URL
    // To do: change the file format to mlpackage
    public init(resourcesAt baseURL: URL) {
        stditURL = baseURL.appending(path: "stdit3.mlmodelc")
        decoderURL = baseURL.appending(path: "vae.mlmodelc") // 참고: 현재 모델 파일 존재하지 않음.
        configT5URL = baseURL.appending(path: "tokenizer_config.json")
        dataT5URL = baseURL.appending(path: "tokenizer.json")
        embedURL = baseURL.appending(path: "t5embed-tokens.mlmodelc")
        finalNormURL = baseURL.appending(path: "t5final-layer-norm.mlmodelc")
    }
}

public struct SoraPipeline {
  // need to initialize the required models. ex) stdit, vae and so on.
  let TextEncodingT5: TextEncoding?
  let STDit: STDit?
  let VAE: VAEDecoder?
  
  init(resourcesAt baseURL: URL, configuration config: MLModelConfiguration = .init(), reduceMemory: Bool = false) throws {
    
    let urls = ResourceURLs(resourcesAt: baseURL)
    
    // initialize Models for Text Encoding
    if FileManager.default.fileExists(atPath: urls.configT5URL.path),
       FileManager.default.fileExists(atPath: urls.dataT5URL.path),
       FileManager.default.fileExists(atPath: urls.embedURL.path),
       FileManager.default.fileExists(atPath: urls.finalNormURL.path),
       FileManager.default.fileExists(atPath: urls.decoderURL.path),
       FileManager.default.fileExists(atPath: urls.configT5URL.path)
    {
      let config = MLModelConfiguration()
      config.computeUnits = .all
      let tokenizerT5 = try PreTrainedTokenizer(tokenizerConfig: Config(fileURL: urls.configT5URL), tokenizerData: Config(fileURL: urls.dataT5URL))
      let embedLayer = ManagedMLModel(modelURL: urls.embedURL, config: config)
      let finalNormLayer = ManagedMLModel(modelURL: urls.finalNormURL, config: config)
      var DivT5s: [ManagedMLModel] = []
      for i in 0...23 {
        let T5BlockURL = baseURL.appending(path: "t5block-layer\(i).mlmodelc")
        DivT5s.append(ManagedMLModel(modelURL: T5BlockURL, config: config))
      }
      TextEncodingT5 = TextEncoding(tokenizer: tokenizerT5, DivT5s: DivT5s, embed: embedLayer, finalNorm: finalNormLayer)
    } else {
      TextEncodingT5 = nil
    }
    
    // initialize Models for STDit
    if FileManager.default.fileExists(atPath: urls.stditURL.path) {
      // To do: STDit Model
      STDit = nil
    } else {
      STDit = nil
    }
    
    // initialize Models for VAE
    if FileManager.default.fileExists(atPath: urls.decoderURL.path) {
      // To do: VAE for decoding video
      VAE = VAEDecoder(modelURL: urls.decoderURL, config: config)
    } else {
      VAE = nil
    }
  }
  func sample(prompt: String) {
    // To do: make the sample process
    do {
      guard let ids = try TextEncodingT5?.tokenize(prompt) else {
        print("Error: Can't tokenize")
        return
      }
      print("Result of Tokenizing: \(ids)")
      guard let resultEncoding = try TextEncodingT5?.encode(ids: ids) else {
        print("Error: Can't Encoding")
        return
      }
      // To do : STDit and VAE
      print("Begin Decoding")
      
      // Sampling loop, here.
      
      
      // Get sample
      let latentShape = [2,3,8,16,16]
      let totalElements = latentShape.reduce(1,*)
      var latentVars = (0..<totalElements).map { _ in Float32(1.0)}
      
      guard let resultDecoding = try VAE?.decode(latentVars: latentVars) else {
        print("Error: Can't Decode")
        return
      }
    } catch {
      print("Error: Can't make sample.")
      print(error)
    }
  }
}


