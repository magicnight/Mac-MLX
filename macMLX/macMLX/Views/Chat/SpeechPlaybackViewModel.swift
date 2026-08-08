// SpeechPlaybackViewModel.swift
// macMLX
//
// Drives the per-message "read this aloud" button.
//
// One instance serves the whole transcript rather than one per bubble, because
// the invariant that matters is global: at most one message is ever being
// spoken. Holding the state centrally makes takeover (clicking a second
// message while the first is still talking) a state transition instead of
// cross-talk between two independent view models that cannot see each other.

import Foundation
import MacMLXCore

@Observable
@MainActor
final class SpeechPlaybackViewModel {

    /// Which message, if any, owns the speaker — and how far along it is.
    /// Every case carries the message id so the view can ask about a specific
    /// bubble rather than tracking ownership itself.
    enum State: Equatable {
        case idle
        case synthesizing(messageID: UUID)
        case speaking(messageID: UUID)
        case failed(messageID: UUID, message: String)
    }

    // MARK: - State

    private(set) var state: State = .idle

    /// Whether `messageID` is the one currently working or speaking — drives
    /// that bubble's button between "speak" and "stop".
    func isActive(_ messageID: UUID) -> Bool {
        switch state {
        case .synthesizing(let id), .speaking(let id): return id == messageID
        case .idle, .failed: return false
        }
    }

    /// Whether `messageID` is waiting on the model, so the button can show
    /// progress rather than a stop glyph over silence.
    func isSynthesizing(_ messageID: UUID) -> Bool {
        state == .synthesizing(messageID: messageID)
    }

    /// The error for `messageID`, or `nil`. Scoped per message so a failure on
    /// one bubble is not reported on every other one.
    func errorMessage(for messageID: UUID) -> String? {
        if case .failed(let id, let message) = state, id == messageID { return message }
        return nil
    }

    // MARK: - Private

    private let service: any AudioServicing
    private let player: any SpeechPlaying
    private var task: Task<Void, Never>?

    init(service: any AudioServicing, player: any SpeechPlaying) {
        self.service = service
        self.player = player
    }

    // MARK: - Actions

    /// The speaker button. Starts `messageID` talking, or stops it if it
    /// already is — and takes over cleanly when a DIFFERENT message is active.
    ///
    /// - Parameter modelID: The TTS repo id, or `nil` when the choice was
    ///   ambiguous or nothing is installed. Reported, never guessed.
    func toggle(
        messageID: UUID, text: String, modelID: String?, voice: String?
    ) async {
        // Second click on the message that owns the speaker: stop.
        if isActive(messageID) {
            stop()
            return
        }

        // A different message owns it — end that one before starting this.
        // Without this, its `onFinish` could arrive mid-synthesis and reset
        // the state the new message just claimed.
        stop()

        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            state = .failed(messageID: messageID, message: Self.emptyTextMessage)
            return
        }
        guard let modelID else {
            state = .failed(messageID: messageID, message: Self.noModelMessage)
            return
        }

        state = .synthesizing(messageID: messageID)

        let service = self.service
        let task = Task { @MainActor [weak self] in
            do {
                let speech = try await service.synthesize(
                    text: trimmed, modelID: modelID, voice: voice)
                guard let self, !Task.isCancelled else { return }
                // Reuse the encoder the HTTP `/v1/audio/speech` path already
                // uses rather than writing a second one. A nil here means the
                // model reported a rate the container cannot describe — rare,
                // but a real failure the user should see rather than a button
                // that quietly does nothing.
                guard let wav = WAVEncoder.encode(
                    samples: speech.samples, sampleRate: speech.sampleRate)
                else {
                    self.state = .failed(
                        messageID: messageID, message: Self.unencodableMessage)
                    return
                }
                // Claim `.speaking` BEFORE handing the bytes over: a player
                // that completes synchronously inside `play` would otherwise
                // run the completion while the state still said `.synthesizing`,
                // whose guard would decline to settle it — leaving the button
                // stuck on a message that had already finished.
                self.state = .speaking(messageID: messageID)
                do {
                    try self.player.play(wav) { [weak self] in
                        // Natural end only — `stop()` clears this first, so a
                        // takeover never resets the newer message's state.
                        guard let self, self.state == .speaking(messageID: messageID) else {
                            return
                        }
                        self.state = .idle
                    }
                } catch {
                    self.state = .failed(
                        messageID: messageID, message: error.localizedDescription)
                }
            } catch {
                guard let self, !Task.isCancelled else { return }
                if error is CancellationError { return }
                self.state = .failed(messageID: messageID, message: error.localizedDescription)
            }
        }
        self.task = task
        await task.value
    }

    /// Silence whatever is playing and abandon any in-flight synthesis.
    /// Idempotent, and safe to call when nothing is active.
    func stop() {
        task?.cancel()
        task = nil
        player.stop()
        // A displayed error is not something `stop()` owns — leaving it up
        // means a failure stays readable until the user acts again.
        if case .failed = state { return }
        state = .idle
    }

    // MARK: - Messages

    static let noModelMessage =
        "No text-to-speech model selected. Install one, or pick which to use in Settings "
        + "if you have more than one."

    static let emptyTextMessage = "Nothing to read out."

    static let unencodableMessage =
        "The model returned audio macMLX could not package for playback."
}
