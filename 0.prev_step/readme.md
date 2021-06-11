# Step 0.* check fetatures 

This directory contains files to check the fit of the mfcc/spectogram to detect phonemes.

This files are unneeded for the rest of the project.


# iOS 

To generate the mfcc/spectogram from iOS, use the app `MarlaMFCC` inside `marla-ios` project.

In order to read 8bit wav files, use the `readFileFloat` method to read the file as Floats. Then transform each value of the buffer using the following algorithm:

    sig = sig * ((2 ** (16 - 1) + 1)

iOS code using `c\_speech\_features`:

    //
    //  main.swift
    //  MarlaMFCC
    //
    //  Created by Daniel Vela Angulo on 08/12/2019.
    //  Copyright © 2019 veladan. All rights reserved.
    //
    import AVFoundation
    import Foundation
    
    func readFileFloat(_ inputFile: String) -> [Float] {
      let url = URL(fileURLWithPath: inputFile)
      guard let audioFile = try? AVAudioFile(forReading: url) else { fatalError() }
      guard let format = AVAudioFormat(commonFormat: .pcmFormatFloat32,
                                       sampleRate: 16000,
                                       channels: 1,
                                       interleaved: false) else { fatalError() }
      let size: UInt32 = 14672
      var output: [Float] = []
      guard let buf = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: size) else { fatalError() }
      do {
      repeat {
        try audioFile.read(into: buf)
        let floatBuffer = UnsafeBufferPointer(start: buf.floatChannelData?.pointee, count: Int(buf.frameLength))
        output.append(contentsOf: Array(floatBuffer))
        print("step")
        } while true
      } catch {
      }
      return output
    }
    
    func writeCSV(_ file: String, _ buffer: [Float], _ nframes: Int) {
      let fileURL = URL(fileURLWithPath: file)
      let elementsPerFrame = buffer.count / nframes;
      var index = 0
      var str = ""
      for _ in 0..<nframes {
        str = str + "\(buffer[index])"
        index += 1
        for _ in 1..<elementsPerFrame {
          str = str + ",\(buffer[index])"
          index += 1
        }
        str = str + "\n"
      }
      try? str.write(to: fileURL, atomically: true, encoding: .utf8)
    }
    
    if CommandLine.arguments.count != 2 {
      print("Usage: MarlaMFCC <input_wav_file>")
      exit(0)
    }
    
    let inputWavFile = CommandLine.arguments[1]
    
    let bufferFloat = readFileFloat(inputWavFile)
    
    // Process float buffer to convert to a Int16 buffer
    let nb_bits = 16
    let max_nb_bit = Float(truncating: pow(2, nb_bits - 1) as NSNumber) + 1.0
    var buffer: [Int16] = []
    for i in 0..<bufferFloat.count {
      let value = Int16(bufferFloat[i] * max_nb_bit)
      buffer.append(value)
    }
    
    // Process MFCC
    var unsafeBuffer = UnsafeMutablePointer<Int16>(mutating: buffer)
    var usafeResult: UnsafeMutablePointer<Float>?
    let usafeResultFrames: Int = Int(run_mfcc(unsafeBuffer, UInt32(buffer.count), &usafeResult));
    let buffSize = usafeResultFrames * 26;
    let uBuffer = UnsafeBufferPointer(start: usafeResult, count: Int(buffSize))
    let finalBuffer = Array(uBuffer)
    let a = 234
    
    let file = inputWavFile + ".MFCC.iOS.csv"
    writeCSV(file, finalBuffer, usafeResultFrames)




