//
//  TextEncoder5Model.swift
//  On-Device-OpenSora
//
//  Created by hanbitchan on 8/14/24.
//

import Foundation
import Tokenizers
import CoreML

public protocol TextEncoderT5Model: ResourceManaging {
    func encode(_ text: String) throws -> TextEncoderT5Output
}

public struct TextEncoderT5Output {
    public let encoderHiddenStates: MLShapedArray<Float32>
}

public struct TextEncoderT5: TextEncoderT5Model {
  var tokenizer: Tokenizer
  var model: ManagedMLModel
  public init(tokenizer: Tokenizer, modelURL: URL, config: MLModelConfiguration) {
    self.tokenizer = tokenizer
    self.model = ManagedMLModel(modelURL: modelURL, config: config)
  }
  
  public func loadResources() throws {
    try model.loadResources()
  }
  public func unloadResources() {
    model.unloadResources()
  }
  
  /// Encode input text/string
  ///
  ///  - Parameters:
  ///     - text: Input text to be tokenized and then embedded
  ///  - Returns: Embedding representing the input text
  public func encode(_ text: String) throws -> TextEncoderT5Output {

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

      // Use the model to generate the embedding
      let encodedText = try encode(ids: ids)
      return encodedText
  }

  func encode(ids: [Int]) throws -> TextEncoderT5Output {
      let inputName = "input_ids"
      let inputShape = inputShape
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

      let inputArray = MLShapedArray<Float32>(scalars: floatIds, shape: inputShape)
      let maskArray = MLShapedArray<Float32>(scalars: floatMask, shape: inputShape)
      let inputFeatures = try! MLDictionaryFeatureProvider(
          dictionary: [inputName: MLMultiArray(inputArray),
                       attentionMaskName: MLMultiArray(maskArray)])

      let result = try model.perform { model in
        // To do: 쪼개진 T5로 돌아가는 pipeline만들기
          try model.prediction(from: inputFeatures)
      }

      let embeddingFeature = result.featureValue(for: "encoder_hidden_states")
      return TextEncoderT5Output(encoderHiddenStates: MLShapedArray<Float32>(converting: embeddingFeature!.multiArrayValue!))
  }

  var inputDescription: MLFeatureDescription {
      try! model.perform { model in
          model.modelDescription.inputDescriptionsByName.first!.value
      }
  }
  
  var inputShape: [Int] {
      inputDescription.multiArrayConstraint!.shape.map { $0.intValue }
  }
  
}
