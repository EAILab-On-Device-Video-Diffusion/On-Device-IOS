//
//  Tensor2Vid.swift
//  tensor_to_video
//
//  Created by ijeong on 10/1/24.
//

import Foundation
import CoreML
import AVFoundation
import VideoToolbox
import SwiftUI

import Foundation
import CoreML
import AVFoundation
import VideoToolbox

@MainActor
final class Tensor2Vid: ObservableObject {
    @Published var videoURL: URL?

    func generateRandomInput(frameCount: Int, width: Int, height: Int) -> MLMultiArray? {
        let shape: [NSNumber] = [NSNumber(value: frameCount), 3, NSNumber(value: height), NSNumber(value: width)]
        guard let multiArray = try? MLMultiArray(shape: shape, dataType: .float32) else {
            print("Failed to create MLMultiArray")
            return nil
        }
        
        for i in 0..<multiArray.count {
            multiArray[i] = NSNumber(value: Float.random(in: 0...1))
        }
        
        return multiArray
    }
  

    func convertToVideo(multiArray: MLMultiArray) async -> URL? {
        let frameCount = multiArray.shape[0].intValue
        let channels = multiArray.shape[1].intValue
        let height = multiArray.shape[2].intValue
        let width = multiArray.shape[3].intValue
        
        guard channels == 3 else {
        print("Invalid number of channels. Expected 3, got \(channels)")
            return nil
        }
        
        let outputURL = FileManager.default.temporaryDirectory.appendingPathComponent("output.mp4")
        let videoWriter: AVAssetWriter
        do {
        videoWriter = try AVAssetWriter(outputURL: outputURL, fileType: .mp4)
        } catch {
        print("Failed to create AVAssetWriter: \(error)")
            return nil
        }
        
        let videoSettings: [String: Any] = [
        AVVideoCodecKey: AVVideoCodecType.h264,
        AVVideoWidthKey: width,
        AVVideoHeightKey: height
        ]
        
        let videoWriterInput = AVAssetWriterInput(mediaType: .video, outputSettings: videoSettings)
        let adaptor = AVAssetWriterInputPixelBufferAdaptor(assetWriterInput: videoWriterInput, sourcePixelBufferAttributes: nil)
        
        videoWriter.add(videoWriterInput)
        videoWriter.startWriting()
        videoWriter.startSession(atSourceTime: .zero)
        
        let frameDuration = CMTimeMake(value: 1, timescale: 30)
        var frameTime = CMTime.zero
        
        for frameIndex in 0..<frameCount {
            autoreleasepool {
                guard let pixelBuffer = createPixelBuffer(from: multiArray, frameIndex: frameIndex, width: width, height: height) else {
                print("Failed to create pixel buffer for frame \(frameIndex)")
                return
                }
                
                while !videoWriterInput.isReadyForMoreMediaData {
                Thread.sleep(forTimeInterval: 0.1)
                }
                
                if adaptor.append(pixelBuffer, withPresentationTime: frameTime) {
                frameTime = CMTimeAdd(frameTime, frameDuration)
                } else {
                print("Failed to append pixel buffer for frame \(frameIndex)")
                }
            }
        }
        
        videoWriterInput.markAsFinished()
        await videoWriter.finishWriting()
        
        await MainActor.run{
            self.videoURL = outputURL
        }
        print("Video saved to: \(outputURL.path)")
        
        
        return outputURL
    }

    private func createPixelBuffer(from multiArray: MLMultiArray, frameIndex: Int, width: Int, height: Int) -> CVPixelBuffer? {
        var pixelBuffer: CVPixelBuffer?
        let attrs = [kCVPixelBufferCGImageCompatibilityKey: kCFBooleanTrue,
                kCVPixelBufferCGBitmapContextCompatibilityKey: kCFBooleanTrue] as CFDictionary
        let status = CVPixelBufferCreate(kCFAllocatorDefault,
                                    width,
                                    height,
                                    kCVPixelFormatType_32BGRA,
                                    attrs,
                                    &pixelBuffer)

        guard status == kCVReturnSuccess, let pixelBuffer = pixelBuffer else {
            return nil
        }

        CVPixelBufferLockBaseAddress(pixelBuffer, CVPixelBufferLockFlags(rawValue: 0))
        let pixelData = CVPixelBufferGetBaseAddress(pixelBuffer)

        let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
        _ = CVPixelBufferGetDataSize(pixelBuffer)

        for y in 0..<height {
            for x in 0..<width {
                let pixelOffset = y * bytesPerRow + x * 4
                guard let pixelOffsetPointer = pixelData?.advanced(by: pixelOffset) else { continue }
                
                let pixel = pixelOffsetPointer.bindMemory(to: UInt8.self, capacity: 4)
                
                let r = UInt8(max(0, min(255, multiArray[[frameIndex, 0, y, x] as [NSNumber]].floatValue * 255)))
                let g = UInt8(max(0, min(255, multiArray[[frameIndex, 1, y, x] as [NSNumber]].floatValue * 255)))
                let b = UInt8(max(0, min(255, multiArray[[frameIndex, 2, y, x] as [NSNumber]].floatValue * 255)))
                
                pixel[0] = b
                pixel[1] = g
                pixel[2] = r
                pixel[3] = 255  // Alpha channel
            }
        }

        CVPixelBufferUnlockBaseAddress(pixelBuffer, CVPixelBufferLockFlags(rawValue: 0))

        return pixelBuffer
    }
}


