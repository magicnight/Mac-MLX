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
// Each member forwards to the engine's model-bound call, which loads the model
// if it is not resident and runs on the model that load produced. That matters
// here: the view models cancel a superseded task and start the next one at
// once, and the superseded call can still be inside the engine — suspended in
// a load, or synthesizing — so a "load, then run on whatever is resident" pair
// could run the new request on the old model, or on nothing.

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
        try await transcribe(model: modelID, audioURL: audioURL, language: nil, temperature: nil).text
    }

    func synthesize(
        text: String, modelID: String, voice: String?
    ) async throws -> AudioEngine.Speech {
        try await synthesize(model: modelID, text: text, voice: voice, language: nil)
    }
}
