// RecordingSpeechPlayer.swift
// macMLXTests
//
// Stand-in for `AVSpeechPlayer` that records what it was asked to play and
// hands the test the completion instead of an audio device.
//
// Holding `onFinish` rather than invoking it is the point: playback in the real
// player ends whenever the audio does, which a test cannot wait for. Parking
// the completion lets a test keep a message "speaking" for as long as it needs
// to observe takeover and stop, then end it on command.

import Foundation
@testable import macMLX

@MainActor
final class RecordingSpeechPlayer: SpeechPlaying {

    enum Failure: Error, Equatable {
        case cannotOpenAudio
    }

    // MARK: - Script

    /// Whether the next `play` throws instead of starting.
    var nextPlayFails = false

    /// Invoke the completion synchronously from inside `play`, the way a real
    /// player would for audio that ends the instant it starts (a one-sample
    /// buffer, say). Off by default because holding the completion is what
    /// most tests need — but the synchronous case is the one that catches an
    /// implementation claiming "speaking" only AFTER handing the bytes over.
    var finishesSynchronously = false

    // MARK: - Recorded calls

    private(set) var played: [Data] = []
    private(set) var stopCount = 0

    var playCount: Int { played.count }
    var lastPlayed: Data? { played.last }

    /// Completion for the most recent successful `play`, or `nil` once it has
    /// been consumed or cleared by `stop()`.
    private(set) var pendingFinish: (@MainActor () -> Void)?

    // MARK: - SpeechPlaying

    func play(_ wav: Data, onFinish: @escaping @MainActor () -> Void) throws {
        if nextPlayFails {
            nextPlayFails = false
            throw Failure.cannotOpenAudio
        }
        played.append(wav)
        if finishesSynchronously {
            onFinish()
            return
        }
        pendingFinish = onFinish
    }

    func stop() {
        stopCount += 1
        // Mirrors `AVSpeechPlayer`: a stop must never be reported as a natural
        // finish, so the completion is dropped rather than invoked.
        pendingFinish = nil
    }

    // MARK: - Test control

    /// Fire the parked completion, as the real player does when audio reaches
    /// its end. No-op when nothing is parked.
    func finishPlayback() {
        let completion = pendingFinish
        pendingFinish = nil
        completion?()
    }
}
