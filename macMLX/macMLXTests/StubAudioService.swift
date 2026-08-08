// StubAudioService.swift
// macMLXTests
//
// Scriptable stand-in for `AudioEngine` on the `AudioServicing` seam. It never
// loads a model: each call returns a canned result, optionally parked behind an
// `AsyncGate` and optionally throwing.
//
// Gates and failures are captured PER CALL, at the moment the method is
// entered, for the same reason `StubBenchmarkEngine` does it: a test that
// overlaps two runs has to be able to script them differently — park the first
// behind gate A, re-arm, then let the second run through.

import Foundation
import MacMLXCore
@testable import macMLX

@MainActor
final class StubAudioService: AudioServicing {

    enum Failure: Error, Equatable {
        case transcriptionFailed
        case synthesisFailed
    }

    // MARK: - Script

    /// Text the next successful transcription returns.
    var transcript = "the quick brown fox"

    /// Samples + rate the next successful synthesis returns. The default is a
    /// short non-silent buffer at a rate `WAVEncoder` can describe.
    var speech = AudioEngine.Speech(samples: [0.0, 0.5, -0.5, 1.0], sampleRate: 24_000)

    /// Gates handed to the next calls, in call order. A call made when the
    /// queue is empty runs straight through.
    var transcribeGates: [AsyncGate] = []
    var synthesizeGates: [AsyncGate] = []

    /// Whether the NEXT call of each kind throws. One-shot: consumed when the
    /// call is entered, so a retry after a scripted failure succeeds unless
    /// the test re-arms it. That is what lets "clicking again after an error
    /// retries" be tested at all.
    var nextTranscribeFails = false
    var nextSynthesizeFails = false

    // MARK: - Recorded calls

    private(set) var transcribeCalls: [(audioURL: URL, modelID: String)] = []
    private(set) var synthesizeCalls: [(text: String, modelID: String, voice: String?)] = []

    var transcribeCallCount: Int { transcribeCalls.count }
    var synthesizeCallCount: Int { synthesizeCalls.count }

    // MARK: - AudioServicing

    func transcribe(audioURL: URL, modelID: String) async throws -> String {
        transcribeCalls.append((audioURL, modelID))
        let gate = transcribeGates.isEmpty ? nil : transcribeGates.removeFirst()
        let fails = nextTranscribeFails
        nextTranscribeFails = false
        if let gate { await gate.wait() }
        if fails { throw Failure.transcriptionFailed }
        return transcript
    }

    func synthesize(
        text: String, modelID: String, voice: String?
    ) async throws -> AudioEngine.Speech {
        synthesizeCalls.append((text, modelID, voice))
        let gate = synthesizeGates.isEmpty ? nil : synthesizeGates.removeFirst()
        let fails = nextSynthesizeFails
        nextSynthesizeFails = false
        if let gate { await gate.wait() }
        if fails { throw Failure.synthesisFailed }
        return speech
    }

    // A note on what this stub deliberately does NOT do: it never calls
    // `Task.checkCancellation()`. That is faithful, not lax. `AudioEngine` runs
    // a SYNCHRONOUS MLX forward pass (`sttModel.generate`, and the boxed
    // `SpeechGenerationModel.generate`) with no cancellation check anywhere in
    // it, so cancelling a real transcription does NOT unwind the call — it
    // returns a perfectly good result that nobody wants any more. A stub that
    // threw `CancellationError` here would quietly relieve the view models of
    // the post-await `Task.isCancelled` checks that are the only thing keeping
    // an abandoned result out of the composer, and the tests would still pass
    // with those checks deleted.
}
