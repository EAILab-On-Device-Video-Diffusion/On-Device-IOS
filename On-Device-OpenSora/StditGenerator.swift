//
//  TextGenerator.swift
//  On-Device-NLP
//
//  Created by 이규환 on 6/25/24.
//

import Foundation
import CoreML

@MainActor
final class StditGenerator: ObservableObject {
  
  func modelProcessing() {
    Task.detached(priority: .high) {
      do {
        let config = MLModelConfiguration()
        config.computeUnits = .cpuOnly
        
        print("Start model processing...")
        let x = try MLMultiArray(shape: [2, 4, 16, 3, 3], dataType: .float32)
        let timestep = try MLMultiArray(shape: [2], dataType: .float32)
        let y = try MLMultiArray(shape: [2, 1, 200, 4096], dataType: .float32)
        

        func randomFloat(mean: Float32, variance: Float32) -> Float32 {
            let u1 = Float32.random(in: 0...1)
            let u2 = Float32.random(in: 0...1)
            let z0 = sqrt(-2.0 * log(u1)) * cos(2.0 * .pi * u2)
            return mean + z0 * sqrt(variance)
        }
        
        let mean = 0.0
        let variance = 1.0
        
        for i in 0..<x.count {
          x[i] = NSNumber(value: randomFloat(mean: Float32(mean), variance: Float32(variance)))
        }
        
        for i in 0..<timestep.count {
          timestep[i] = NSNumber(value: 999.0)
        }
        
        for i in 0..<y.count {
          y[i] = NSNumber(value: randomFloat(mean: Float32(mean), variance: Float32(variance)))
        }

        if let vmodel = try? stdit3(configuration: config) {
          print("Start Stdit")
          let vinput = stdit3Input(input: x, timestep: timestep, y: y)
          let vresult = try await vmodel.prediction(input: vinput).var_14604
          let c = vresult.count
          print(c)
          
          var out = [Float]()
          print(vresult[10])
          for i in 0..<c {
            out.append(vresult[i].floatValue)
          }
          print(out)
          print("Done stdit")
        }
      } catch {
        print(error)
      }
    }
        
  }
}

