//
//  MultiModalDiffusionTransformer.swift
//  On-Device-OpenSora
//
//  Created by hanbitchan on 8/14/24.
//

import CoreML

public struct STDit3 {
  var spatialAndTemporals: [ManagedMLModel]
  init(spatialAndTemporals: [ManagedMLModel]) {
    self.spatialAndTemporals = spatialAndTemporals
  }
  
  func sample(x:MLShapedArray<Float32>, timestep: MLShapedArray<Float32>, modelargs:Dictionary<String, MLShapedArray<Float32>>) async throws -> MLTensor {

    // === Start layer ===

    let config = MLModelConfiguration()
    config.computeUnits = .cpuAndGPU
    var stdit3Part1: stdit3_part1? = try! stdit3_part1(configuration: config)
    let stdit3Part1Input = stdit3_part1Input(z_in: x, t: timestep, y: modelargs["y"]!, mask: modelargs["mask"]!, x_mask: modelargs["x_mask"]!, fps: modelargs["fps"]!, height: modelargs["height"]!, width: modelargs["width"]!)
    let stdit3Part1Output = try await stdit3Part1!.prediction(input: stdit3Part1Input)
    print("=== Done stdit3 Part1 ===")
    stdit3Part1 = nil
    
    var inputFeatures = try MLDictionaryFeatureProvider (
      dictionary : [ "x" : stdit3Part1Output.featureValue(for: "x"), "y" : stdit3Part1Output.featureValue(for: "outY"), "y_lens": stdit3Part1Output.featureValue(for: "y_lens"), "mask_expanded": stdit3Part1Output.featureValue(for: "mask_expanded") ,"t_mlp": stdit3Part1Output.featureValue(for: "t_mlp"), "x_mask": stdit3Part1Output.featureValue(for: "x_mask"), "t0_mlp": stdit3Part1Output.featureValue(for: "t0_mlp"), "T": stdit3Part1Output.featureValue(for: "T"), "S": stdit3Part1Output.featureValue(for: "S")]
    )
    
    // === blocks ===
    for (i, spatialAndTemporal) in spatialAndTemporals.enumerated() {
      let spatialOutput = try spatialAndTemporal.perform { model in
        try model.prediction(from: inputFeatures)
      }
      spatialAndTemporal.unloadResources()
      inputFeatures = try MLDictionaryFeatureProvider (
        dictionary : [ "x" : spatialOutput.featureValue(for: "output"), "y" : stdit3Part1Output.featureValue(for: "outY"), "y_lens": stdit3Part1Output.featureValue(for: "y_lens"), "mask_expanded": stdit3Part1Output.featureValue(for: "mask_expanded"), "t_mlp": stdit3Part1Output.featureValue(for: "t_mlp"), "x_mask": stdit3Part1Output.featureValue(for: "x_mask"), "t0_mlp": stdit3Part1Output.featureValue(for: "t0_mlp"), "T": stdit3Part1Output.featureValue(for: "T"), "S": stdit3Part1Output.featureValue(for: "S")]
      )
      print("=== Done stdit3 Spatial & Temporal \(i) ===")
    }
    
    // === final layer ===
    var stdit3Part2: stdit3_part2? = try! stdit3_part2(configuration: config)
    let stdit3Part2Input = stdit3_part2Input(x: (inputFeatures.featureValue(for: "x")?.multiArrayValue!)!, x_mask: (stdit3Part1Output.featureValue(for: "x_mask")?.multiArrayValue!)!, t: (stdit3Part1Output.featureValue(for: "outT")?.multiArrayValue!)!, t0: (stdit3Part1Output.featureValue(for: "t0")?.multiArrayValue!)!, T: (stdit3Part1Output.featureValue(for: "T")?.multiArrayValue!)!, S: (stdit3Part1Output.featureValue(for: "S")?.multiArrayValue!)!, H: (stdit3Part1Output.featureValue(for: "H")?.multiArrayValue!)!, W: (stdit3Part1Output.featureValue(for: "W")?.multiArrayValue!)!, Tx: (stdit3Part1Output.featureValue(for: "Tx")?.multiArrayValue!)!, Hx: (stdit3Part1Output.featureValue(for: "Hx")?.multiArrayValue!)!, Wx: (stdit3Part1Output.featureValue(for: "Wx")?.multiArrayValue!)!)
    let stdit3Part2Output = try await stdit3Part2!.prediction(input: stdit3Part2Input)
    stdit3Part2 = nil
    print("=== Done stdit3 Part2 ===")
    
    let output = MLTensor(stdit3Part2Output.outputShapedArray)
    return output
  }
}
