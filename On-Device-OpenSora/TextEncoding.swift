//
//  TextEncoding.swift
//  On-Device-OpenSora
//
//  Created by hanbitchan on 8/29/24.
//

import Foundation
import Tokenizers
import CoreML
import SwiftUI

//public struct TextEncoderT5Output {
//    public let encoderHiddenStates: MLShapedArray<Float32>
//}

public struct TextEncoding {
  var tokenizer: Tokenizer
  var DivT5s: [ManagedMLModel]
  var embed: ManagedMLModel
  var dropout: ManagedMLModel
  var finalNorm: ManagedMLModel
  
  init(tokenizer: Tokenizer, DivT5s: [ManagedMLModel], embed: ManagedMLModel, dropout: ManagedMLModel, finalNorm: ManagedMLModel) {
    self.tokenizer = tokenizer
    self.DivT5s = DivT5s
    self.embed = embed
    self.dropout = dropout
    self.finalNorm = finalNorm
  }
  
  func tokenize(_ text: String) throws -> [Int] {
    // Get models expected input length
    let inputLength = inputShape.last!
    // Tokenize, padding to the expected length
    var tokens = tokenizer.tokenize(text: text)
    var ids = tokens.map { tokenizer.convertTokenToId($0) ?? 0 }
    // Truncate if necessary
    if ids.count > inputLength {
        tokens = tokens.dropLast(tokens.count - inputLength)
        ids = ids.dropLast(ids.count - inputLength)
        print("Needed to truncate input for TextEncoderT5")
    }
    print("Done tokenizing")
    return ids
  }
  
  func encode(ids: [Int]) throws -> TextEncoderT5Output {
    let startT5Time = DispatchTime.now()

    let inputName = "input_ids"
//    let inputShape = inputShape
    let inputShape = [1,300, 4096]
    let inputLength = inputShape[1]
            
    let bosToken = tokenizer.bosTokenId ?? 0
    let eosToken = tokenizer.eosTokenId ?? 1
    let padToken = bosToken
    let maskToken = eosToken

    // Truncate and pad input to the expected length
    let truncatedIds = ids.prefix(inputLength - 1) + [eosToken]
    let inputIds = truncatedIds + Array(repeating: padToken, count: inputLength - truncatedIds.count)

    let attentionMaskName = "attention_mask"
    var attentionMask: [Int] = inputIds.map { token in
        token == padToken ? maskToken : padToken
    }
    attentionMask[0] = bosToken

    let floatIds = inputIds.map { Float32($0) }
    let floatMask = attentionMask.map { Float32($0) }
    

    let inputShapeEmbed = inputShapeEmbed
    let inputArrayEmbed = MLShapedArray<Float32>(scalars: floatIds, shape: inputShapeEmbed)
    let inputFeaturesEmbed = try! MLDictionaryFeatureProvider(dictionary: [inputName: MLMultiArray(inputArrayEmbed)])
    
    let resultEmbed = try embed.perform { model in
      try model.prediction(from: inputFeaturesEmbed)
    }
    let inputFeaturesDropout1 = try! MLDictionaryFeatureProvider(dictionary: ["input": resultEmbed.featureValue(for: "var_6")])
    print("Done Embedding")
    let resultDropout1 = try dropout.perform { model in
      try model.prediction(from: inputFeaturesDropout1)
    }
    print("Done First Dropout")
    
    let maskArray = MLShapedArray<Float32>(scalars: floatMask, shape: [1,1,1,300])
    var inputFeatures: MLFeatureProvider = try! MLDictionaryFeatureProvider(
        dictionary: ["hidden_states": resultDropout1.featureValue(for: "cast_0"),
                     attentionMaskName: MLMultiArray(maskArray)])
    print("Processing DivT5s")
    for (index,model) in DivT5s.enumerated() {
      let layerOutputs = try model.perform { model in
        try model.prediction(from: inputFeatures)
      }
      model.unloadResources()
      if index == 0 {
          inputFeatures = try! MLDictionaryFeatureProvider(
            dictionary: ["hidden_states": layerOutputs.featureValue(for: "var_167"),
                         attentionMaskName: MLMultiArray(maskArray), "position_bias" : layerOutputs.featureValue(for: "position_bias")])
      } else {
          inputFeatures = try! MLDictionaryFeatureProvider(
            dictionary: ["hidden_states": layerOutputs.featureValue(for: "var_119"),
                         attentionMaskName: MLMultiArray(maskArray), "position_bias" : layerOutputs.featureValue(for: "cast_2")])
      }
      print("Done T5_layer_\(index)_Block")
    }
    print("Done DivT5s")
    let hidden_states = inputFeatures.featureValue(for: "hidden_states")
    
    let inputFeaturesNorm = try! MLDictionaryFeatureProvider(dictionary: ["input": hidden_states])

    let resultNorm = try finalNorm.perform { model in
      try model.prediction(from: inputFeaturesNorm)
    }
    print("Done Final Norm")
      
    let inputFeaturesDropout2 = try! MLDictionaryFeatureProvider(dictionary: ["input": resultNorm.featureValue(for: "var_9")])

    let resultDropout2 = try dropout.perform { model in
      try model.prediction(from: inputFeaturesDropout2)
    }
    print("Done Second Dropout")
    print("Done T5 processing")
    let endT5Time = DispatchTime.now()
    let elapsedT5Time = endT5Time.uptimeNanoseconds - startT5Time.uptimeNanoseconds
    print("T5 Running Time: \(Double(elapsedT5Time) / 1000000000)")
    let count = resultDropout2.featureValue(for: "cast_0")?.multiArrayValue?.count
    var output = [Float]()
      for i in 0..<count! {
          output.append(resultDropout2.featureValue(for: "cast_0")?.multiArrayValue![i] as! Float)
      }
    print(output[0...10])
    return TextEncoderT5Output(encoderHiddenStates: MLShapedArray<Float32>(converting: resultDropout2.featureValue(for: "cast_0")!.multiArrayValue!))
  }
  
  var inputDescription: MLFeatureDescription {
      try! DivT5s[0].perform { model in
          model.modelDescription.inputDescriptionsByName.first!.value
      }
  }
  
  var inputDescriptionEmbed: MLFeatureDescription {
    try! embed.perform { model in
        model.modelDescription.inputDescriptionsByName.first!.value
    }
  }
  
  var inputShape: [Int] {
      inputDescription.multiArrayConstraint!.shape.map { $0.intValue }
  }
  
  var inputShapeEmbed: [Int] {
    inputDescriptionEmbed.multiArrayConstraint!.shape.map { $0.intValue}
  }
}


