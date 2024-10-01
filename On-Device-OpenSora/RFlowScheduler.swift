//
//  RFlowScheduler.swift
//  On-Device-OpenSora
//
//  Created by hanbitchan on 9/30/24.
//

import Accelerate
import CoreML

/* 주석은 인퍼런스 시 필요없는 부분임. */

public final class RFlowScheduler {
  public let numTimesteps: Int
//  public let numSamplingsteps: Int
//  public let useDiscreteTimestepTransform: Bool
//  public let useTimestepTransform: Bool
//  public let sample_method: String = "uniform"
//  public let loc: Float = 0.0
//  public let scale: Float = 1.0
//  public let use_timestep_transform: Bool = false
//  public let transform_scale: Float = 1.0
  
  public init(numTimesteps: Int /*numSamplingsteps: Int, useDiscreteTimestepTransform: Bool, useTimestepTransform:Bool*/) {
    self.numTimesteps = numTimesteps
//    self.numSamplingsteps = numSamplingsteps
//    self.useDiscreteTimestepTransform = useDiscreteTimestepTransform
//    self.useTimestepTransform = useTimestepTransform
  }
  
  public func addNoise(original_samples:MLTensor, noise: MLTensor, timesteps: Float) -> MLTensor {
    var timepoints = MLTensor([1.0 - Float(timesteps) / Float(self.numTimesteps)])
    
    timepoints = timepoints.expandingShape(at: 1).expandingShape(at: 1).expandingShape(at: 1).expandingShape(at: 1)
//    timepoints = timepoints.repeat(1, noise.shape[1], noise.shape[2], noise.shape[3], noise.shape[4])

    return timepoints * original_samples + (1.0 - timepoints) * noise
    
  }
}
