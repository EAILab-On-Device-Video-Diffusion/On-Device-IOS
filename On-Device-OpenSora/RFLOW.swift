//
//  Scheduler.swift
//  On-Device-OpenSora
//
//  Created by hanbitchan on 8/14/24.
//

import Accelerate
import CoreML
import Foundation

/* 주석은 인퍼런스 시 필요없는 부분임. */

public struct RFLOWInput {
  let model: STDit3
  let modelArgs: Dictionary<String, MLShapedArray<Float32>> // T5에서 나온 놈.
  let z: MLShapedArray<Float32>
  let mask: MLShapedArray<Float32>
  let additionalArgs: [String: MLTensor]
  let BDM: MLShapedArray<Float32>
  let resolution: Double
}

public final class RFLOW {
  
  public let numSamplingsteps: Int
  public let numTimesteps: Int
  public let cfgScale: Float
  public let scheduler: RFlowScheduler
//  public let useDiscreteTimestepTransform: Bool
//  public let useTimestepTransform: Bool
  
  public init(numSamplingsteps: Int = 10, numTimesteps: Int = 1000, cfgScale: Float = 4.0 /*, useDiscreteTimestepTransform: Bool = false, useTimestepTransform: Bool = false*/) {
    self.numSamplingsteps = numSamplingsteps
    self.numTimesteps = numTimesteps
    self.cfgScale = cfgScale
    self.scheduler = RFlowScheduler(numTimesteps: numTimesteps)
//    self.useDiscreteTimestepTransform = useDiscreteTimestepTransform
//    self.useTimestepTransform = useTimestepTransform
  }
  
  public func sample(rflowInput: RFLOWInput, yNull: MLShapedArray<Float32>) async -> MLTensor {
    let guidanceScale = self.cfgScale
    
    // text encoding
    var modelArgs = rflowInput.modelArgs
    // null_y = self.y_embedder.y_embedding[None].repeat(n, 1, 1)[:, None]
    modelArgs["y"] = MLShapedArray(concatenating: [modelArgs["y"]!, yNull], alongAxis: 0)
    for (key, value) in rflowInput.additionalArgs {
      modelArgs[key] = await value.shapedArray(of: Float32.self)
    }
    modelArgs["BDM"] = rflowInput.BDM
    // prepare timesteps
    var timeSteps: [Float32] = []
    for i in 0..<self.numSamplingsteps {
      var t = (1.0 - Float32(i) / Float32(self.numSamplingsteps)) * Float32(self.numTimesteps)
      t = timestep_transform(t: round(t), num_timesteps: self.numTimesteps, resolution: rflowInput.resolution)
      timeSteps.append(t)
    }
    
    let mask = MLTensor(rflowInput.mask)
    var noiseAdded = MLTensor(repeating: false, shape: mask.shape, scalarType: Bool.self)
    
    noiseAdded = noiseAdded .| (mask .== 1)
    let numTimestepsTensor = MLTensor(repeating: Float32(self.numTimesteps), shape: mask.shape)
    var z = MLTensor(rflowInput.z)
    let startTime = DispatchTime.now()

    for (i,t) in timeSteps.enumerated() {
      print("== Step \(i) ==")
      // mask for adding noise
      let mask_t = mask * numTimestepsTensor
      let x0 = z
      let xNoise = self.scheduler.addNoise(original_samples: x0, noise: MLTensor(randomNormal: x0.shape, seed: 42,scalarType: Float32.self), timesteps: t)
      
      let T = MLTensor([Float32(t)])
      let maskTUpper = mask_t .>= T.expandingShape(at: 1)
      modelArgs["x_mask"] = await MLTensor(concatenating: [maskTUpper, maskTUpper], alongAxis: 0).cast(to: Float32.self).shapedArray(of: Float32.self)

      let maskAddNoise: MLTensor? = maskTUpper .& .!noiseAdded
      // z = torch.where(maskAddNoise[:, None, :, None, None], x_noise, x0) 대안 코드
      let expandedMaskAN = maskAddNoise!.cast(to: Float32.self).expandingShape(at: 1).expandingShape(at: 3).expandingShape(at: 4)
      z = expandedMaskAN * xNoise + (1.0 - expandedMaskAN) * x0
      
      noiseAdded = maskTUpper
      
      // classifier-free guidance
      let zIn = await MLTensor(concatenating: [z,z], alongAxis: 0).shapedArray(of: Float32.self)
      let tIn = await MLTensor(concatenating: [T,T], alongAxis: 0).shapedArray(of: Float32.self)
//      var pred: MLTensor = try! await rflowInput.model.sample(x: zIn, timestep: tIn, modelargs: modelArgs)
      var pred: MLTensor = try! await rflowInput.model.sample(x: zIn, timestep: tIn, modelargs: modelArgs)
//      print(await pred.shapedArray(of: Float32.self))
//      var printOutput = [Float32]()
//      for i in 0...100 {
//        printOutput.append(Float32(MLMultiArray(await pred.shapedArray(of: Float32.self))[i]))
//        }
//      print(printOutput[0...100])

      let splitSize1 = pred.shape[1] / 2
      pred = pred.split(sizes: [splitSize1,splitSize1], alongAxis: 1)[0] //chuck
//      printOutput = []
//      for i in 0...100 {
//        printOutput.append(Float32(MLMultiArray(await pred.shapedArray(of: Float32.self))[i]))
//        }
//
//      print("pred")
//      print(printOutput[0...100])

      let splitSize2 = pred.shape[0] / 2
      let finalPred = pred.split(sizes: [splitSize2, splitSize2], alongAxis: 0) // chuck
      let vPred = finalPred[1] + guidanceScale * (finalPred[0] - finalPred[1])

      // update z
      var dt = i < timeSteps.count - 1 ? timeSteps[i] - timeSteps[i + 1] : timeSteps[i]
      dt = Float32(dt) / Float32(self.numTimesteps)
      let DT = MLTensor([dt])
      
      // z = z + v_pred * dt[:, None, None, None, None] 대안코드
      z = z + vPred * DT.expandingShape(at: 1).expandingShape(at: 2).expandingShape(at: 3).expandingShape(at: 4)
      
      // z = torch.where(mask_t_upper[:, None, :, None, None], z, x0) 대안 코드
      let expandedMaskTU = maskTUpper.cast(to: Float32.self).expandingShape(at: 1).expandingShape(at: 3).expandingShape(at: 4)
      z = expandedMaskTU * z + (1.0 - expandedMaskTU) * x0
//      print(await z.shapedArray(of: Float32.self))
    }
    let endTime = DispatchTime.now()
    let elapsedTime = endTime.uptimeNanoseconds - startTime.uptimeNanoseconds
    print("STDit Running Time: \(Double(elapsedTime) / 1000000000)")
    return z
  }
}

extension RFLOW {
  func timestep_transform(t: Float32, num_timesteps: Int = 1, resolution: Double) -> Float32 {
    let base_resolution = 512.0 * 512.0
    
    let T = t / Float32(num_timesteps)
    let ratio_space = Float32(resolution / base_resolution).squareRoot()
    let ratio_time = Float32(Int(51 / 17) * 5).squareRoot()
    let ratio = ratio_space * ratio_time
    var new_t = ratio * T / (1 + (ratio - 1) * T)
    new_t = new_t * Float32(num_timesteps)
    return new_t
  }
}
