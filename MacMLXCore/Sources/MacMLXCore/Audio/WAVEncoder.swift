import Foundation

/// Encodes float audio samples into a canonical RIFF/WAVE container carrying
/// 16-bit signed little-endian PCM — the one audio format macMLX can produce
/// honestly on Apple platforms without a third-party encoder.
///
/// Pure and synchronous by design: no Metal, no model, no file system. The
/// byte layout is the whole contract, so it is unit-tested directly.
///
/// Layout produced (44-byte canonical header, then interleaved samples):
///
/// ```text
/// offset  size  field
///  0      4     "RIFF"
///  4      4     chunk size          = 36 + dataBytes      (UInt32 LE)
///  8      4     "WAVE"
/// 12      4     "fmt "
/// 16      4     fmt chunk size      = 16                  (UInt32 LE)
/// 20      2     audio format        = 1 (PCM)             (UInt16 LE)
/// 22      2     channels                                  (UInt16 LE)
/// 24      4     sample rate                               (UInt32 LE)
/// 28      4     byte rate = rate * channels * 2           (UInt32 LE)
/// 32      2     block align = channels * 2                (UInt16 LE)
/// 34      2     bits per sample     = 16                  (UInt16 LE)
/// 36      4     "data"
/// 40      4     data size = sampleCount * 2               (UInt32 LE)
/// 44      …     PCM samples                               (Int16 LE)
/// ```
public enum WAVEncoder {

    /// Bytes in the canonical PCM header this encoder emits.
    public static let headerByteCount = 44

    /// Bits per sample in every container this encoder emits.
    public static let bitsPerSample = 16

    /// Convert float samples in `[-1, 1]` to raw 16-bit little-endian PCM with
    /// NO container — the headerless byte stream OpenAI's `pcm` response
    /// format is defined as.
    ///
    /// Out-of-range inputs are CLIPPED (not wrapped): anything at or beyond
    /// `+1` saturates to `Int16.max`, anything at or beyond `-1` saturates to
    /// `Int16.min`. Non-finite samples (NaN / ±inf) are treated as silence
    /// rather than producing undefined integer conversions.
    public static func pcm16LittleEndian(samples: [Float]) -> Data {
        var out = Data(capacity: samples.count * 2)
        for sample in samples {
            let scaled: Int16
            if sample.isNaN {
                scaled = 0
            } else if sample >= 1.0 {
                scaled = Int16.max
            } else if sample <= -1.0 {
                scaled = Int16.min
            } else {
                // 32767 (not 32768) so the positive side can never overflow on
                // rounding; the asymmetry is the conventional audio mapping.
                scaled = Int16((sample * 32767.0).rounded())
            }
            let bits = UInt16(bitPattern: scaled)
            out.append(UInt8(bits & 0xFF))
            out.append(UInt8(bits >> 8))
        }
        return out
    }

    /// Wrap float samples in a complete WAV file.
    ///
    /// - Parameters:
    ///   - samples: Interleaved float samples in `[-1, 1]`; clipped as
    ///     documented on ``pcm16LittleEndian(samples:)``.
    ///   - sampleRate: Frames per second. Must be positive.
    ///   - channels: Interleaved channel count. Must be positive.
    /// - Returns: The complete file bytes, or `nil` when `sampleRate` /
    ///   `channels` are not positive, or when the payload cannot be described
    ///   by the 32-bit RIFF size fields.
    public static func encode(samples: [Float], sampleRate: Int, channels: Int = 1) -> Data? {
        guard sampleRate > 0, channels > 0 else { return nil }
        let bytesPerSample = bitsPerSample / 8
        let dataByteCount = samples.count * bytesPerSample
        // RIFF sizes are UInt32. Guard every field that has to hold one.
        let riffChunkSize = 36 + dataByteCount
        guard dataByteCount <= Int(UInt32.max), riffChunkSize <= Int(UInt32.max),
              sampleRate <= Int(UInt32.max), channels <= Int(UInt16.max)
        else { return nil }
        let byteRate = sampleRate * channels * bytesPerSample
        let blockAlign = channels * bytesPerSample
        guard byteRate <= Int(UInt32.max), blockAlign <= Int(UInt16.max) else { return nil }

        var out = Data(capacity: headerByteCount + dataByteCount)
        out.append(contentsOf: Array("RIFF".utf8))
        appendUInt32LE(&out, UInt32(riffChunkSize))
        out.append(contentsOf: Array("WAVE".utf8))
        out.append(contentsOf: Array("fmt ".utf8))
        appendUInt32LE(&out, 16)                       // PCM fmt chunk size
        appendUInt16LE(&out, 1)                        // format tag: PCM
        appendUInt16LE(&out, UInt16(channels))
        appendUInt32LE(&out, UInt32(sampleRate))
        appendUInt32LE(&out, UInt32(byteRate))
        appendUInt16LE(&out, UInt16(blockAlign))
        appendUInt16LE(&out, UInt16(bitsPerSample))
        out.append(contentsOf: Array("data".utf8))
        appendUInt32LE(&out, UInt32(dataByteCount))
        out.append(pcm16LittleEndian(samples: samples))
        return out
    }

    // MARK: Private

    private static func appendUInt32LE(_ data: inout Data, _ value: UInt32) {
        data.append(UInt8(value & 0xFF))
        data.append(UInt8((value >> 8) & 0xFF))
        data.append(UInt8((value >> 16) & 0xFF))
        data.append(UInt8((value >> 24) & 0xFF))
    }

    private static func appendUInt16LE(_ data: inout Data, _ value: UInt16) {
        data.append(UInt8(value & 0xFF))
        data.append(UInt8((value >> 8) & 0xFF))
    }
}
