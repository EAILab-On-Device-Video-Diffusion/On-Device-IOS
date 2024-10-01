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
  let model: STDit
  let modelArgs: Dictionary<String, MLShapedArray<Float32>> // T5에서 나온 놈.
  let z: MLShapedArray<Float32>
  let mask: MLShapedArray<Float32>
  let additionalArgs: [String: MLTensor]
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
  
  public func sample(rflowInput: RFLOWInput) async -> MLTensor {
    let guidanceScale = self.cfgScale
    let n = 1 // 이 방식으로 해도 될까?
    
    // text encoding
    var modelArgs = rflowInput.modelArgs
    let yNull = try? MLShapedArray<Float32>(MLMultiArray(shape: [NSNumber(value: n), 1], dataType: .float32))
    modelArgs["y"] = MLShapedArray(concatenating: [modelArgs["y"]!, yNull!], alongAxis: 0)
    
    for (key, value) in rflowInput.additionalArgs {
      modelArgs[key] = await value.shapedArray(of: Float32.self)
    }
    
    // prepare timesteps
    var timeSteps: [Float] = []
    for i in 0..<numTimesteps {
      let t = (1.0 - Float(i) / Float(self.numSamplingsteps)) * Float(self.numTimesteps)
      timeSteps.append(t)
    }
    var maskShape: [Int] = []
    for i in 0..<rflowInput.mask.shape.count {
      maskShape.append(Int(truncating: rflowInput.mask.shape[i] as NSNumber))
    }
    
    let mask = MLTensor(rflowInput.mask)
    var noiseAdded = MLTensor(zeros: maskShape, scalarType: Float32.self)
    
    noiseAdded = noiseAdded .| (mask .== 1)
    let numTimestepsTensor = MLTensor(repeating: Float(self.numTimesteps), shape: maskShape)
    var z = MLTensor(rflowInput.z)
    
    for (i,t) in timeSteps.enumerated() {
      // mask for adding noise
      let mask_t = mask * numTimestepsTensor
      let x0 = z
      let xNoise = self.scheduler.addNoise(original_samples: x0, noise: MLTensor(randomNormal: x0.shape, scalarType: Float32.self), timesteps: t)
      var T = MLTensor([Float32(t)])
      let maskTUpper = mask_t .>= T.expandingShape(at: 1)
      modelArgs["x_mask"] = await MLTensor(concatenating: [maskTUpper, maskTUpper], alongAxis: 0).shapedArray(of: Float32.self) // 안될수도?
      let maskAddNoise = maskTUpper .& .!noiseAdded

      // z = torch.where(maskAddNoise[:, None, :, None, None], x_noise, x0) 대안 코드
      let expandedMaskAN = maskAddNoise.expandingShape(at: 1).expandingShape(at: 3).expandingShape(at: 4)
      z = expandedMaskAN * xNoise + (1 - expandedMaskAN) * x0
      
      noiseAdded = maskTUpper

      // classifier-free guidance
      let zIn = await MLTensor(concatenating: [z,z], alongAxis: 0).shapedArray(of: Float32.self)
      T = MLTensor(concatenating: [T,T], alongAxis: 0)
      
      // To Do: chuck 구현 필요, input으로 들어갈 때는 shapedArray로 들어가야함
      var pred: MLTensor = rflowInput.model.sample(x: zIn, timestep: Float32(t), modelargs: modelArgs)
      let splitSize1 = pred.shape[1] / 2
      pred = pred.split(sizes: [splitSize1,splitSize1], alongAxis: 1)[0]
      
      let splitSize2 = pred.shape[0] / 2
      let finalPred = pred.split(sizes: [splitSize2, splitSize2], alongAxis: 0)
      let vPred = finalPred[1] + guidanceScale * (finalPred[0] - finalPred[1])

      // update z
      var dt = i < timeSteps.count - 1 ? timeSteps[i] - timeSteps[i + 1] : timeSteps[i]
      dt = dt / Float(self.numTimesteps)
      let DT = MLTensor([dt])
      
      // z = z + v_pred * dt[:, None, None, None, None] 대안코드
      z = z + DT.expandingShape(at: 1).expandingShape(at: 2).expandingShape(at: 3).expandingShape(at: 4)
      
      // z = torch.where(mask_t_upper[:, None, :, None, None], z, x0) 대안 코드
      let expandedMaskTU = maskTUpper.expandingShape(at: 1).expandingShape(at: 3).expandingShape(at: 4)
      z = expandedMaskTU * z + (1 - expandedMaskTU) * x0
    }
    return z
  }
}

