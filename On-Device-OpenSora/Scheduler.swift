//
//  Scheduler.swift
//  On-Device-OpenSora
//
//  Created by hanbitchan on 8/14/24.
//

import Accelerate
import CoreML

public final class RFlowScheduler {
  public let timesteps: Int
  public let sampling_steps: Int
  public let discrete_timesteps: Bool
  public let sample_method: String = "uniform"
  public let loc: Float = 0.0
  public let scale: Float = 1.0
  public let use_timestep_transform: Bool = false
  public let transform_scale: Float = 1.0
  
  public init(timesteps: Int, sampling_steps: Int, discrete_timesteps: Bool) {
    self.timesteps = timesteps
    self.sampling_steps = sampling_steps
    self.discrete_timesteps = discrete_timesteps
  }
}
