// AudioTranscriptionViewModel.swift
// macMLX
//
// Drives the chat input's audio attachment: pick a file, transcribe it with
// the resident STT model, and drop the text into the composer for the user to
// edit before sending.
//
// Text lands in the composer rather than being sent straight off, because
// speech recognition is wrong often enough that auto-sending would make the
// user's first interaction with the feature a correction. Same reason the
// VLM image path stages attachments instead of firing on drop.

import Foundation
import MacMLXCore

@Observable
@MainActor
final class AudioTranscriptionViewModel {

    /// Where a transcription can be. Errors carry their message so the view
    /// never has to map a case back to prose — and so a test can assert the
    /// user is actually told something specific.
    enum State: Equatable {
        case idle
        case transcribing
        case failed(message: String)
    }

    // MARK: - State

    private(set) var state: State = .idle

    var isTranscribing: Bool { state == .transcribing }

    /// Message to show, or `nil` when there is nothing wrong.
    var errorMessage: String? {
        if case .failed(let message) = state { return message }
        return nil
    }

    // MARK: - Private

    private let service: any AudioServicing
    private var task: Task<Void, Never>?

    init(service: any AudioServicing) {
        self.service = service
    }

    // MARK: - Actions

    /// Transcribe `audioURL` and hand the text to `sink`.
    ///
    /// `sink` runs ONLY on success with non-empty text — every other outcome
    /// leaves the composer untouched and puts a message in ``state``. Nothing
    /// here fails silently: a user who clicks the button always sees either
    /// text appear or a reason it did not.
    ///
    /// - Parameter modelID: The STT repo id, or `nil` when the choice was
    ///   ambiguous or nothing is installed (see
    ///   `LocalModel.resolveAudioModelID`). `nil` is reported to the user, not
    ///   guessed around.
    func transcribe(
        audioURL: URL,
        modelID: String?,
        sink: @escaping @MainActor (String) -> Void
    ) async {
        guard let modelID else {
            state = .failed(message: Self.noModelMessage)
            return
        }

        // Supersede any in-flight transcription — the user picked a new file,
        // so the older result is stale by definition. Same cancel-then-replace
        // discipline as `ModelLibraryViewModel.loadLocalModels`.
        task?.cancel()
        state = .transcribing

        let service = self.service
        let task = Task { @MainActor [weak self] in
            do {
                let text = try await service.transcribe(audioURL: audioURL, modelID: modelID)
                guard let self, !Task.isCancelled else { return }
                let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
                guard !trimmed.isEmpty else {
                    // A real outcome (silence, or audio with no speech), not an
                    // error from the engine — but appending nothing and calling
                    // it done would look identical to a broken button.
                    self.state = .failed(message: Self.noSpeechMessage)
                    return
                }
                self.state = .idle
                sink(trimmed)
            } catch {
                guard let self, !Task.isCancelled else { return }
                if error is CancellationError { return }
                self.state = .failed(message: error.localizedDescription)
            }
        }
        self.task = task
        await task.value
    }

    /// Abandon an in-flight transcription. The composer is left untouched and
    /// no error is shown — the user asked for this.
    func cancel() {
        task?.cancel()
        task = nil
        if state == .transcribing { state = .idle }
    }

    /// Dismiss a displayed error without starting anything new.
    func clearError() {
        if case .failed = state { state = .idle }
    }

    // MARK: - Messages

    static let noModelMessage =
        "No speech-to-text model selected. Install one, or pick which to use in Settings "
        + "if you have more than one."

    static let noSpeechMessage = "No speech found in that file."
}
