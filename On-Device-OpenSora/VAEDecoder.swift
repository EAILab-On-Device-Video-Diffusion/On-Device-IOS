//
//  Decoder.swift
//  On-Device-OpenSora
//
//  Created by hanbitchan on 8/14/24.
//

import Foundation
import CoreML

public struct VAEDecoder: ResourceManaging {
  
  var model: ManagedMLModel
  
  public init(modelURL: URL, config: MLModelConfiguration) {
    self.model = ManagedMLModel(modelURL: modelURL, config: config)
  }
  
  public func loadResources() throws {
    try model.loadResources()
  }
  
  public func unloadResources() {
    model.unloadResources()
  }
  // To do: func decode
  
}
