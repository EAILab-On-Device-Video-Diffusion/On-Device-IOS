//
//  MultiModalDiffusionTransformer.swift
//  On-Device-OpenSora
//
//  Created by hanbitchan on 8/14/24.
//

import CoreML

public struct STDit: ResourceManaging {
  var models: [ManagedMLModel]
  
  public init(modelURL: URL, config: MLModelConfiguration) {
    self.models = [ManagedMLModel(modelURL: modelURL, config: config)]
  }
  public func loadResources() throws {
    for model in models {
      try model.loadResources()
    }
  }
  public func unloadResources() {
    for model in models {
      model.unloadResources()
    }
  }
  // To do: Sample

}
