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
  
  enum DecoderError: Error {
      case noDecodedData
  }

  func decode(latentVars: [Float32]) throws -> MLMultiArray {
      let startVAEDecodeTime = DispatchTime.now()

      // Define input shape for the decoder
      let latentShape = [1, 4, 4, 20, 27] // Your latent dimension
      let totalElements = latentShape.reduce(1, *) // Calculate total number of elements

      // Check if the size of latentVars matches the expected total elements
      guard latentVars.count == totalElements else {
          throw DecoderError.noDecodedData
      }

      // Calculate the sum of the input latent variables
      let sumOfInput = latentVars.reduce(0, +)
      print("Sum of Input Latent Variables: \(sumOfInput)")
      
    
    // Create an MLShapedArray from the latentVars
      let latentArray = MLShapedArray<Float32>(scalars: latentVars, shape: latentShape)
      let numFrames = MLShapedArray(arrayLiteral: 20)

      // Prepare input features with only latent input
      let inputFeatures = try MLDictionaryFeatureProvider(dictionary: [
        "latents": MLMultiArray(latentArray), "num_frames": MLMultiArray(numFrames) // Adjust input name as necessary
      ])
      print("Begin Decoding")
    
      // Pass inputs through the decoder model
      let decoderOutput = try model.perform { model in
          try model.prediction(from: inputFeatures)
      }

      print("Done Decoding")

      // Log time and return output
      let endVAEDecodeTime = DispatchTime.now()
      let elapsedVAEDecodeTime = endVAEDecodeTime.uptimeNanoseconds - startVAEDecodeTime.uptimeNanoseconds
      print("VAE Decode Running Time: \(Double(elapsedVAEDecodeTime) / 1000000000) seconds")
      // Extract the result from the decoder's output
      let outputFeatureName = "var_4026" // Replace with the actual output feature name in your model
      let decodedValues = decoderOutput.featureValue(for: outputFeatureName)?.multiArrayValue

      // Check if decodedValues is nil or empty
      guard let decodedValues = decodedValues, decodedValues.count > 0 else {
          print("No decoded data found.")
          throw DecoderError.noDecodedData
      }

      print(type(of: decodedValues), "; type of decodedValues")

      return decodedValues
  }

  
    
    
}
