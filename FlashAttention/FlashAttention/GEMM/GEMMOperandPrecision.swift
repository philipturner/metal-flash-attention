//
//  GEMMOperandPrecision.swift
//  FlashAttention
//
//  Created by Philip Turner on 6/21/24.
//

/// An enumeration of the precisions supported by the kernel.
///
/// If you wish to support quantized precisions, copy/translate the source code
/// and integrate a modified version into your app. Something similar to a Swift
/// `enum` (e.g. C++ `enum class`) could enumerate the quantization formats
/// used by application code. An exemplary set could be:
/// - FP32
/// - FP16
/// - BF16
/// - signed 8-bit integer
/// - s1ezm7
/// - FP8
/// - palletized
///
/// If you support non-floating-point formats, you have the responsibility of
/// authoring correct and performant GPU code for them. A general rule of thumb,
/// is keep the data compressed in `device` or `threadgroup` memory. Transform
/// into a floating point type while loading into the registers. Keep the
/// accumulator in floating point until the output needs to be written.
/// If the output is quantized, it will be compressed when writing back to
/// `device` memory (or `threadgroup` before the async copy in edge cases).
///
/// For example, the reference implementation treats BF16 like a quantized
/// integer type on Apple7 and Apple8 GPUs. It is decompressed to FP32 in
/// registers.
enum GEMMOperandPrecision: UInt16 {
  case FP32 = 0
  case FP16 = 1
  case BF16 = 2
}
