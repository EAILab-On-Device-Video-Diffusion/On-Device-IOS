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
  public func sample(x:MLShapedArray<Float32>, timestep: Float32, modelargs:Dictionary<String, MLShapedArray<Float32>>) -> MLTensor {
    
    // 임시
    let result = MLTensor(1)
    return result
  }
}
