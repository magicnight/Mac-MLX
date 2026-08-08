// SpeechPlaying.swift
// macMLX
//
// The audio-output seam behind `SpeechPlaybackViewModel`.
//
// Separate from `AudioServicing` because it is a separate dependency with a
// separate reason to be stubbed: synthesis is model work, playback is
// AVFoundation. Tests need to drive the two independently — in particular to
// hold a "playing" state open and then decide when it ends, which is what
// makes the takeover and stop invariants observable at all.

import Foundation

@MainActor
protocol SpeechPlaying: AnyObject {

    /// Begin playing `wav`, invoking `onFinish` when it plays to its end.
    ///
    /// `onFinish` fires ONLY for natural completion. A ``stop()`` — whether
    /// from the user or from another message taking over — must not call it,
    /// so the view model can settle its own state without racing a callback
    /// from the run it just abandoned.
    ///
    /// - Throws: When the bytes cannot be opened for playback. The caller
    ///   surfaces this rather than leaving a button stuck mid-state.
    func play(_ wav: Data, onFinish: @escaping @MainActor () -> Void) throws

    /// Stop any in-flight playback. Idempotent; never invokes `onFinish`.
    func stop()
}
