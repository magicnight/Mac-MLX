// AVSpeechPlayer.swift
// macMLX
//
// The production `SpeechPlaying`: an `AVAudioPlayer` over the WAV bytes
// `SpeechPlaybackViewModel` hands it.
//
// `AVAudioPlayer` rather than `AVAudioEngine`: the input is a complete,
// in-memory PCM buffer that `WAVEncoder` already produced, so there is nothing
// to schedule or mix and the simpler API is the honest fit.
//
// No `AVAudioSession` configuration — that type is iOS-only; on macOS a plain
// `AVAudioPlayer` routes to the default output device with no setup, and this
// app deliberately never touches audio INPUT (mic capture is out of scope until
// the code-signing story is settled, so the app requests no microphone
// authorization and declares no `NSMicrophoneUsageDescription`).

import AVFoundation
import Foundation

@MainActor
final class AVSpeechPlayer: NSObject, SpeechPlaying {

    private var player: AVAudioPlayer?

    /// Completion for the CURRENT playback. Cleared by `stop()` before the
    /// player is torn down, which is what keeps a stop from being reported as
    /// a natural finish.
    private var onFinish: (@MainActor () -> Void)?

    func play(_ wav: Data, onFinish: @escaping @MainActor () -> Void) throws {
        stop()
        let player = try AVAudioPlayer(data: wav)
        player.delegate = self
        self.player = player
        self.onFinish = onFinish

        // `play()` can return false after a successful init — no output device,
        // the buffer refuses to prepare. No delegate callback follows a refusal,
        // so ignoring the result strands the caller: it has already entered its
        // speaking state and would wait forever for a finish that cannot come.
        // Tear down and throw so the failure surfaces where the caller can show
        // it, exactly as an init failure does.
        guard player.play() else {
            stop()
            throw PlaybackError.couldNotStart
        }
    }

    enum PlaybackError: LocalizedError {
        case couldNotStart

        var errorDescription: String? {
            switch self {
            case .couldNotStart:
                "Could not start audio playback."
            }
        }
    }

    func stop() {
        // Drop the completion FIRST: `AVAudioPlayer.stop()` does not fire the
        // delegate callback, but clearing it up front also covers the reentrant
        // case where `play` stops a previous run.
        onFinish = nil
        player?.stop()
        player = nil
    }
}

// MARK: - AVAudioPlayerDelegate

extension AVSpeechPlayer: AVAudioPlayerDelegate {

    nonisolated func audioPlayerDidFinishPlaying(
        _ player: AVAudioPlayer, successfully flag: Bool
    ) {
        // Delivered on the thread that started playback; hop to the main actor
        // where all of this class's state (and the view model's) lives.
        Task { @MainActor [weak self] in
            guard let self else { return }
            let completion = self.onFinish
            self.onFinish = nil
            self.player = nil
            completion?()
        }
    }

    nonisolated func audioPlayerDecodeErrorDidOccur(
        _ player: AVAudioPlayer, error: (any Error)?
    ) {
        // A mid-stream decode failure ends this playback. Settle to idle the
        // same way a natural finish does rather than leaving the button stuck
        // showing "speaking" over a player that is no longer running.
        Task { @MainActor [weak self] in
            guard let self else { return }
            let completion = self.onFinish
            self.onFinish = nil
            self.player = nil
            completion?()
        }
    }
}
