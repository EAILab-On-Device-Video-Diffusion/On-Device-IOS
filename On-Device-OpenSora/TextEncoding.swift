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
  var finalNorm: ManagedMLModel
  
  init(tokenizer: Tokenizer, DivT5s: [ManagedMLModel], embed: ManagedMLModel, finalNorm: ManagedMLModel) {
    self.tokenizer = tokenizer
    self.DivT5s = DivT5s
    self.embed = embed
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
    let inputShape = [1,512, 4096]
    let inputLength = inputShape[1]
            
    let bosToken = tokenizer.bosTokenId ?? 0
    let eosToken = tokenizer.eosTokenId ?? 1
    let padToken = bosToken
    let maskToken = -Float32.greatestFiniteMagnitude
    let truncatedIds = ids.prefix(inputLength - 1) + [eosToken]
    let inputIds = truncatedIds + Array(repeating: padToken, count: inputLength - truncatedIds.count)

    var attentionMask: [Float32] = inputIds.map { token in
      token == padToken ? maskToken : 0.0
    }
    attentionMask[0] = 0.0

    let floatIds = inputIds.map { Float32($0) }

    let inputShapeEmbed = inputShapeEmbed
    let inputArrayEmbed = MLShapedArray<Float32>(scalars: floatIds, shape: inputShapeEmbed)
    let inputFeaturesEmbed = try! MLDictionaryFeatureProvider(dictionary: [inputName: MLMultiArray(inputArrayEmbed)])
    
    let resultEmbed = try embed.perform { model in
      try model.prediction(from: inputFeaturesEmbed)
    }
    embed.unloadResources()

    print("Done Embedding")
    
    let maskArray = MLShapedArray<Float32>(scalars: attentionMask, shape: [1,1,1,512])
    var inputFeatures: MLFeatureProvider = try! MLDictionaryFeatureProvider(
      dictionary: ["hidden_states": resultEmbed.featureValue(for: "output") as Any,
                     "attention_mask": MLMultiArray(maskArray)])

    print("Processing DivT5s")
    
    for (index,model) in DivT5s.enumerated() {
      let layerOutputs = try model.perform { model in
        try model.prediction(from: inputFeatures)
      }
      model.unloadResources()
      if index == 0 {
          inputFeatures = try! MLDictionaryFeatureProvider(
            dictionary: ["hidden_states": layerOutputs.featureValue(for: "output_hidden_states") as Any,
                         "attention_mask": MLMultiArray(maskArray),
                         "position_bias" : layerOutputs.featureValue(for: "output_position_bias") as Any])
      } else {
          inputFeatures = try! MLDictionaryFeatureProvider(
            dictionary: ["hidden_states": layerOutputs.featureValue(for: "var_119") as Any,
                         "attention_mask": MLMultiArray(maskArray),
                         "position_bias" : layerOutputs.featureValue(for: "position_bias") as Any])
      }
//      var output = [Float]()
//      for i in 0...100 {
//            output.append(inputFeatures.featureValue(for: "hidden_states")?.multiArrayValue![i] as! Float)
//        }
//      print(output[0...100])
      print("Done T5_layer_\(index)_Block")
    }
    print("Done DivT5s")
    let hidden_states = inputFeatures.featureValue(for: "hidden_states")

    let inputFeaturesNorm = try! MLDictionaryFeatureProvider(dictionary: ["input": hidden_states as Any])

    let resultNorm = try finalNorm.perform { model in
      try model.prediction(from: inputFeaturesNorm)
    }
    print("Done Final Norm")
      
    print("Done T5 processing")
    
    // For Debugging
    let endT5Time = DispatchTime.now()
    let elapsedT5Time = endT5Time.uptimeNanoseconds - startT5Time.uptimeNanoseconds
    print("T5 Running Time: \(Double(elapsedT5Time) / 1000000000)")
//    var output2 = [Float]()
//    let prompt_length = truncatedIds.count
//    for var i in 0..<prompt_length {
//      i = i*4096
//      output2.append(resultNorm.featureValue(for: "output")?.multiArrayValue![i] as! Float)
//      output2.append(resultNorm.featureValue(for: "output")?.multiArrayValue![i+1] as! Float)
//      output2.append(resultNorm.featureValue(for: "output")?.multiArrayValue![i+2] as! Float)
//      i += 4096
//      output2.append(resultNorm.featureValue(for: "output")?.multiArrayValue![i-3] as! Float)
//      output2.append(resultNorm.featureValue(for: "output")?.multiArrayValue![i-2] as! Float)
//      output2.append(resultNorm.featureValue(for: "output")?.multiArrayValue![i-1] as! Float)
//    }
//    print(output2)
    return TextEncoderT5Output(encoderHiddenStates: MLShapedArray<Float32>(converting: resultNorm.featureValue(for: "output")!.multiArrayValue!))
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

