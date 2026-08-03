import Foundation
import Testing
@testable import MacMLXCore

/// Pure unit tests for ``WAVEncoder``. The byte layout IS the contract — a
/// client decodes these bytes with no other information — so the header is
/// asserted field by field rather than round-tripped through a decoder that
/// might share the same mistake. No model, no Metal, no I/O.
@Suite("WAVEncoder")
struct WAVEncoderTests {

    // MARK: Readers

    private func ascii(_ data: Data, _ offset: Int, _ count: Int) -> String {
        String(decoding: data[offset..<(offset + count)], as: UTF8.self)
    }

    private func uint32LE(_ data: Data, _ offset: Int) -> UInt32 {
        UInt32(data[offset]) | UInt32(data[offset + 1]) << 8
            | UInt32(data[offset + 2]) << 16 | UInt32(data[offset + 3]) << 24
    }

    private func uint16LE(_ data: Data, _ offset: Int) -> UInt16 {
        UInt16(data[offset]) | UInt16(data[offset + 1]) << 8
    }

    private func int16LE(_ data: Data, _ offset: Int) -> Int16 {
        Int16(bitPattern: uint16LE(data, offset))
    }

    // MARK: Header

    @Test
    func writesCanonicalRIFFWAVEHeader() throws {
        let samples: [Float] = [0, 0.5, -0.5, 0.25]
        let wav = try #require(WAVEncoder.encode(samples: samples, sampleRate: 24_000))

        #expect(wav.count == WAVEncoder.headerByteCount + samples.count * 2)
        #expect(ascii(wav, 0, 4) == "RIFF")
        #expect(ascii(wav, 8, 4) == "WAVE")
        #expect(ascii(wav, 12, 4) == "fmt ")
        #expect(ascii(wav, 36, 4) == "data")

        #expect(uint32LE(wav, 4) == UInt32(36 + samples.count * 2))  // RIFF chunk size
        #expect(uint32LE(wav, 16) == 16)                            // PCM fmt chunk size
        #expect(uint16LE(wav, 20) == 1)                             // format tag: PCM
        #expect(uint16LE(wav, 22) == 1)                             // channels
        #expect(uint32LE(wav, 24) == 24_000)                        // sample rate
        #expect(uint32LE(wav, 28) == 24_000 * 1 * 2)                // byte rate
        #expect(uint16LE(wav, 32) == 2)                             // block align
        #expect(uint16LE(wav, 34) == 16)                            // bits per sample
        #expect(uint32LE(wav, 40) == UInt32(samples.count * 2))     // data chunk size
    }

    @Test
    func headerTracksSampleRateAndChannelCount() throws {
        let wav = try #require(
            WAVEncoder.encode(samples: [0, 0, 0, 0], sampleRate: 44_100, channels: 2))
        #expect(uint16LE(wav, 22) == 2)
        #expect(uint32LE(wav, 24) == 44_100)
        #expect(uint32LE(wav, 28) == 44_100 * 2 * 2)
        #expect(uint16LE(wav, 32) == 4)
    }

    @Test
    func emptyInputProducesAValidHeaderOnlyFile() throws {
        let wav = try #require(WAVEncoder.encode(samples: [], sampleRate: 24_000))
        #expect(wav.count == WAVEncoder.headerByteCount)
        #expect(ascii(wav, 0, 4) == "RIFF")
        #expect(uint32LE(wav, 4) == 36)  // 36 + 0 data bytes
        #expect(uint32LE(wav, 40) == 0)
    }

    // MARK: Invalid parameters

    @Test
    func rejectsNonPositiveSampleRate() {
        #expect(WAVEncoder.encode(samples: [0], sampleRate: 0) == nil)
        #expect(WAVEncoder.encode(samples: [0], sampleRate: -1) == nil)
    }

    @Test
    func rejectsNonPositiveChannelCount() {
        #expect(WAVEncoder.encode(samples: [0], sampleRate: 24_000, channels: 0) == nil)
        #expect(WAVEncoder.encode(samples: [0], sampleRate: 24_000, channels: -2) == nil)
    }

    // MARK: Sample conversion

    @Test
    func mapsFullScaleSamplesToTheInt16Extremes() {
        let pcm = WAVEncoder.pcm16LittleEndian(samples: [1.0, -1.0, 0.0])
        #expect(pcm.count == 6)
        #expect(int16LE(pcm, 0) == Int16.max)
        #expect(int16LE(pcm, 2) == Int16.min)
        #expect(int16LE(pcm, 4) == 0)
    }

    @Test
    func clipsOutOfRangeSamplesInsteadOfWrappingThem() {
        // The failure this guards against is integer wraparound turning a loud
        // sample into the opposite-polarity extreme — audible as a click.
        let pcm = WAVEncoder.pcm16LittleEndian(samples: [3.7, -9.0, 1.0001, -1.0001])
        #expect(int16LE(pcm, 0) == Int16.max)
        #expect(int16LE(pcm, 2) == Int16.min)
        #expect(int16LE(pcm, 4) == Int16.max)
        #expect(int16LE(pcm, 6) == Int16.min)
    }

    @Test
    func treatsNonFiniteSamplesAsSilence() {
        let pcm = WAVEncoder.pcm16LittleEndian(
            samples: [.nan, .infinity, -.infinity, .signalingNaN])
        #expect(int16LE(pcm, 0) == 0)          // NaN
        #expect(int16LE(pcm, 2) == Int16.max)  // +inf clips positive
        #expect(int16LE(pcm, 4) == Int16.min)  // -inf clips negative
        #expect(int16LE(pcm, 6) == 0)          // signaling NaN
    }

    @Test
    func scalesMidScaleSamplesByThe32767Convention() {
        let pcm = WAVEncoder.pcm16LittleEndian(samples: [0.5, -0.5])
        #expect(int16LE(pcm, 0) == Int16((0.5 * 32767.0 as Float).rounded()))
        #expect(int16LE(pcm, 2) == Int16((-0.5 * 32767.0 as Float).rounded()))
    }

    @Test
    func writesSamplesLittleEndian() {
        // 0x1234 little-endian is [0x34, 0x12]. Pick a sample that lands there.
        let target: Int16 = 0x1234
        let sample = Float(target) / 32767.0
        let pcm = WAVEncoder.pcm16LittleEndian(samples: [sample])
        #expect(pcm.count == 2)
        #expect(pcm[0] == 0x34)
        #expect(pcm[1] == 0x12)
    }

    @Test
    func emptySampleArrayProducesNoPCMBytes() {
        #expect(WAVEncoder.pcm16LittleEndian(samples: []).isEmpty)
    }

    @Test
    func containerPayloadIsByteIdenticalToTheHeaderlessEncoding() throws {
        let samples: [Float] = [0.1, -0.2, 0.9, -1.0, 0.0]
        let wav = try #require(WAVEncoder.encode(samples: samples, sampleRate: 16_000))
        let raw = WAVEncoder.pcm16LittleEndian(samples: samples)
        #expect(wav.suffix(from: WAVEncoder.headerByteCount) == raw)
    }
}
