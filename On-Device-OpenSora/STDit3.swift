//
//  MultiModalDiffusionTransformer.swift
//  On-Device-OpenSora
//
//  Created by hanbitchan on 8/14/24.
//

import CoreML

public struct STDit3 {
  var part1: ManagedMLModel
  var spatialAndTemporals: [ManagedMLModel]
  var part2: ManagedMLModel
  init(part1: ManagedMLModel, spatialAndTemporals: [ManagedMLModel], part2: ManagedMLModel) {
    self.part1 = part1
    self.spatialAndTemporals = spatialAndTemporals
    self.part2 = part2
  }
  
  func sample(x:MLShapedArray<Float32>, timestep: MLShapedArray<Float32>, modelargs:Dictionary<String, MLShapedArray<Float32>>) async throws -> MLTensor {
    // === Start layer ===
    var inputFeatures = try MLDictionaryFeatureProvider(
      dictionary: ["z_in": MLMultiArray(x), "t": MLMultiArray(timestep), "y": MLMultiArray(modelargs["y"]!), "mask": MLMultiArray(modelargs["mask"]!), "fps": MLMultiArray(modelargs["fps"]!), "height": MLMultiArray(modelargs["height"]!), "width": MLMultiArray(modelargs["width"]!), "padH": MLMultiArray(modelargs["padH"]!), "padW": MLMultiArray(modelargs["padW"]!)]
    )
    let stdit3Part1Output = try part1.perform { model in
      try model.prediction(from: inputFeatures)
    }
    part1.unloadResources()

    print("=== Done stdit3 Part1 ===")
    
    inputFeatures = try MLDictionaryFeatureProvider (
      dictionary : [ "x" : stdit3Part1Output.featureValue(for: "x")!, "y" : stdit3Part1Output.featureValue(for: "outY")!, "attn": MLMultiArray(modelargs["BDM"]!), "t_mlp": stdit3Part1Output.featureValue(for: "t_mlp")!, "x_mask": MLMultiArray(modelargs["x_mask"]!), "t0_mlp": stdit3Part1Output.featureValue(for: "t0_mlp")!, "T": stdit3Part1Output.featureValue(for: "T")!, "S": stdit3Part1Output.featureValue(for: "S")!]
    )


    // === blocks ===
    let loadQueue = DispatchQueue(label: "modelLoadQueue", qos: .background, attributes: .concurrent)
    let computeQueue = DispatchQueue(label: "modelComputeQueue", qos: .userInitiated)
    for (i, spatialAndTemporal) in spatialAndTemporals.enumerated() {
        let group = DispatchGroup()
        
        group.enter()
        computeQueue.async {
            do {
                let spatialOutput = try spatialAndTemporal.perform { model in
                    try model.prediction(from: inputFeatures)
                }
                spatialAndTemporal.unloadResources()
                inputFeatures = try MLDictionaryFeatureProvider (
                dictionary : [ "x" : spatialOutput.featureValue(for: "output")!, "y" : stdit3Part1Output.featureValue(for: "outY")!, "attn": MLMultiArray(modelargs["BDM"]!), "t_mlp": stdit3Part1Output.featureValue(for: "t_mlp")!, "x_mask": MLMultiArray(modelargs["x_mask"]!), "t0_mlp": stdit3Part1Output.featureValue(for: "t0_mlp")!, "T": stdit3Part1Output.featureValue(for: "T")!, "S": stdit3Part1Output.featureValue(for: "S")!])
                
                } catch {
                    print("Failed to perform prediction")
                }
                group.leave()
        }
//        if i < spatialAndTemporals.count - 1 {
//            let nextModel = spatialAndTemporals[i + 1]
//            //group.enter()
//            loadQueue.async {
//                do {
//                    try nextModel.loadResources()
//                } catch {
//                    print("Failed to load next Model")
//                }
//                //group.leave()
//            }
////        }
        group.wait()
//        let spatialOutput = try spatialAndTemporal.perform { model in
//        try model.prediction(from: inputFeatures)
//        }
//        spatialAndTemporal.unloadResources()
//        inputFeatures = try MLDictionaryFeatureProvider (
//        dictionary : [ "x" : spatialOutput.featureValue(for: "output")!, "y" : stdit3Part1Output.featureValue(for: "outY")!, "attn": MLMultiArray(modelargs["BDM"]!), "t_mlp": stdit3Part1Output.featureValue(for: "t_mlp")!, "x_mask": MLMultiArray(modelargs["x_mask"]!), "t0_mlp": stdit3Part1Output.featureValue(for: "t0_mlp")!, "T": stdit3Part1Output.featureValue(for: "T")!, "S": stdit3Part1Output.featureValue(for: "S")!]
//        )
    }
    
    // === final layer ===
    inputFeatures = try MLDictionaryFeatureProvider(dictionary: ["z_in": MLMultiArray(x), "x": (inputFeatures.featureValue(for: "x")?.multiArrayValue!)!, "x_mask": MLMultiArray(modelargs["x_mask"]!), "t": (stdit3Part1Output.featureValue(for: "outT")?.multiArrayValue!)!, "t0": (stdit3Part1Output.featureValue(for: "t0")?.multiArrayValue!)!, "padH": MLMultiArray(modelargs["padH"]!), "padW": MLMultiArray(modelargs["padW"]!)])
    
    let stdit3Part2Output = try part2.perform { model in
      try model.prediction(from: inputFeatures)
    }
    part2.unloadResources()
    
    let output = MLTensor(MLShapedArray<Float32>((stdit3Part2Output.featureValue(for: "output")?.multiArrayValue)!))
    return output
  }
}
