//
//  Decoder.swift
//  On-Device-OpenSora
//
//  Created by hanbitchan on 8/14/24.
//

import Foundation
import CoreML

public struct VAEDecoder {
  
  var model: ManagedMLModel
  
  public init(modelURL: URL, config: MLModelConfiguration) {
    self.model = ManagedMLModel(modelURL: modelURL, config: config)
  }
  
  enum DecoderError: Error {
      case noDecodedData
  }

  func decode(latentVars: MLShapedArray<Float32>) throws -> MLMultiArray {
      let startVAEDecodeTime = DispatchTime.now()

      // Define input shape for the decoder
//      let latentShape = [1, 4, 15, 20, 27] // Your latent dimension
//      let totalElements = latentShape.reduce(1, *) // Calculate total number of elements

      // Check if the size of latentVars matches the expected total elements
//      guard latentVars.count == totalElements else {
//          throw DecoderError.noDecodedData
//      }

      // Calculate the sum of the input latent variables
//      let sumOfInput = latentVars.reduce(0, +)
//      print("Sum of Input Latent Variables: \(sumOfInput)")
      
    
    // Create an MLShapedArray from the latentVars
//      let latentArray = MLShapedArray<Float32>(scalars: latentVars, shape: latentShape)
      let numFrames = MLShapedArray(arrayLiteral: 51)

      // Prepare input features with only latent input
      let inputFeatures = try MLDictionaryFeatureProvider(dictionary: [
        "latents": latentVars
      ])
      print("Begin Decoding")
    
//      // Pass inputs through the decoder model
//      let decoderOutput = try model.perform { model in
//          try model.prediction(from: inputFeatures)
//      }
      let inputVae = vaeInput(latents: latentVars)
      let modelVae = try vae(configuration: model.config)
      let decoderOutput = try modelVae.prediction(input: inputVae)
      print("Done Decoding")

      // Log time and return output
      let endVAEDecodeTime = DispatchTime.now()
      let elapsedVAEDecodeTime = endVAEDecodeTime.uptimeNanoseconds - startVAEDecodeTime.uptimeNanoseconds
      print("VAE Decode Running Time: \(Double(elapsedVAEDecodeTime) / 1000000000) seconds")
      // Extract the result from the decoder's output
      let decodedValues = decoderOutput.featureValue(for: "output")?.multiArrayValue

      // Check if decodedValues is nil or empty
      guard let decodedValues = decodedValues, decodedValues.count > 0 else {
          print("No decoded data found.")
          throw DecoderError.noDecodedData
      }

      // Convert the MLMultiArray to a Float array
      var output = [Float]()
      for i in 0..<10 {
          output.append(decodedValues[i].floatValue)
      }
      // Print the first 10 decoded values for debugging purposes
      print("Decoded Output (first 10 values): \(output)")
    return decodedValues
  }

  
    
    
}
