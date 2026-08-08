// AudioServicing.swift
// macMLX
//
// The slice of `AudioEngine` the chat GUI actually uses.
//
// Extracted as a protocol purely to give the audio view models an injection
// seam, exactly like `BenchmarkEngineProviding` does for the benchmark: the
// real `AudioEngine` still does the work in production, and each member below
// forwards to the identical engine calls the view model would otherwise have
// made inline.
//
// The two members are coarser than the engine's own API on purpose. `AudioEngine`
// separates loading from inference (`loadSTT` then `transcribe`), but every GUI
// entry point has to do both — and the engine already no-ops a `load` for a
// model that is resident, so pairing them costs nothing and removes an ordering
// mistake the view models would otherwise be able to make.

import Foundation
import MacMLXCore

protocol AudioServicing: Sendable {

    /// Load `modelID` if it is not already resident, then transcribe
    /// `audioURL`, returning just the text.
    ///
    /// - Parameter modelID: A Hugging Face repo id (`owner/name`) — the only
    ///   shape `AudioEngine.loadSTT` accepts. `ModelLibraryManager.scanAudioModels`
    ///   produces `LocalModel.id`s in exactly this shape.
    func transcribe(audioURL: URL, modelID: String) async throws -> String

    /// Load `modelID` if it is not already resident, then synthesize `text`.
    ///
    /// Returns raw samples plus rate rather than an encoded container, so the
    /// caller picks the container — the GUI reuses `WAVEncoder` and never
    /// re-implements one.
    func synthesize(
        text: String, modelID: String, voice: String?
    ) async throws -> AudioEngine.Speech
}

extension AudioEngine: AudioServicing {

    func transcribe(audioURL: URL, modelID: String) async throws -> String {
        try await loadSTT(modelID)
        return try await transcribe(audioURL: audioURL, language: nil, temperature: nil).text
    }

    func synthesize(
        text: String, modelID: String, voice: String?
    ) async throws -> AudioEngine.Speech {
        try await loadTTS(modelID)
        return try await synthesize(text: text, voice: voice, language: nil)
    }
}
