// SpeechPlaybackViewModelTests.swift
// macMLXTests
//
// State-machine invariants for the per-message read-aloud button.
//
// The property that matters throughout is single ownership: at most one
// message is ever speaking, clicking a second one hands the speaker over
// cleanly, and a run that has been superseded can never reach back and
// overwrite the state of the run that replaced it. That last one is an
// ordering property, so the tests park synthesis behind an `AsyncGate` and
// hold the player's completion rather than sleeping.

import Foundation
import Testing
import MacMLXCore
@testable import macMLX

@Suite("SpeechPlaybackViewModel")
@MainActor
final class SpeechPlaybackViewModelTests {

    private let service = StubAudioService()
    private let player = RecordingSpeechPlayer()
    private let viewModel: SpeechPlaybackViewModel
    private let messageA = UUID()
    private let messageB = UUID()

    init() {
        viewModel = SpeechPlaybackViewModel(service: service, player: player)
    }

    private func toggle(
        _ messageID: UUID,
        text: String = "hello there",
        modelID: String? = "mlx-community/Kokoro-82M",
        voice: String? = nil
    ) async {
        await viewModel.toggle(
            messageID: messageID, text: text, modelID: modelID, voice: voice)
    }

    // MARK: - Happy path

    @Test
    func speakingSynthesizesThenPlaysAndReportsActive() async {
        await toggle(messageA)

        #expect(service.synthesizeCallCount == 1)
        #expect(player.playCount == 1)
        #expect(viewModel.state == .speaking(messageID: messageA))
        #expect(viewModel.isActive(messageA))
        #expect(viewModel.isActive(messageB) == false)
    }

    @Test
    func theResolvedModelAndVoiceReachTheEngine() async {
        await toggle(messageA, text: "  spoken text  ", modelID: "some/tts", voice: "af_heart")

        let call = service.synthesizeCalls.first
        #expect(call?.modelID == "some/tts")
        #expect(call?.voice == "af_heart")
        // Trimmed before synthesis — leading whitespace is not something to
        // spend model time on.
        #expect(call?.text == "spoken text")
    }

    /// The bytes handed to the player must be a real WAV container produced by
    /// `WAVEncoder`, not a hand-rolled header: reusing the encoder the HTTP
    /// `/v1/audio/speech` path already uses is the point of the design.
    @Test
    func playerReceivesWAVEncoderOutput() async throws {
        service.speech = AudioEngine.Speech(samples: [0.0, 0.25, -0.25], sampleRate: 16_000)

        await toggle(messageA)

        let played = try #require(player.lastPlayed)
        let expected = try #require(
            WAVEncoder.encode(samples: [0.0, 0.25, -0.25], sampleRate: 16_000))
        #expect(played == expected)
        #expect(played.prefix(4) == Data("RIFF".utf8))
    }

    /// Natural completion settles back to idle so the button returns to
    /// "speak" without the user having to click stop on finished audio.
    @Test
    func naturalFinishReturnsToIdle() async {
        await toggle(messageA)
        #expect(viewModel.isActive(messageA))

        player.finishPlayback()

        #expect(viewModel.state == .idle)
        #expect(viewModel.isActive(messageA) == false)
    }

    /// Audio short enough to end inside `play` must still settle to idle. The
    /// view model has to claim `.speaking` BEFORE handing the bytes over —
    /// claim it after, and the completion runs while the state still says
    /// `.synthesizing`, whose guard declines to settle it, leaving the button
    /// stuck on a message that has already finished.
    @Test
    func playbackThatEndsSynchronouslyStillSettlesToIdle() async {
        player.finishesSynchronously = true

        await toggle(messageA)

        #expect(player.playCount == 1)
        #expect(viewModel.state == .idle)
        #expect(viewModel.isActive(messageA) == false)
    }

    // MARK: - Stop and takeover

    @Test
    func secondClickOnTheSameMessageStops() async {
        await toggle(messageA)
        let stopsBefore = player.stopCount

        await toggle(messageA)

        #expect(player.stopCount > stopsBefore)
        #expect(viewModel.state == .idle)
        // The stop must not have started a second synthesis.
        #expect(service.synthesizeCallCount == 1)
    }

    /// Clicking a different message while one is speaking hands the speaker
    /// over: the first is silenced, the second plays, and only the second is
    /// reported active.
    @Test
    func clickingAnotherMessageTakesOverCleanly() async {
        await toggle(messageA)
        #expect(viewModel.isActive(messageA))

        await toggle(messageB)

        #expect(player.stopCount >= 1)
        #expect(player.playCount == 2)
        #expect(viewModel.state == .speaking(messageID: messageB))
        #expect(viewModel.isActive(messageA) == false)
        #expect(viewModel.isActive(messageB))
    }

    /// The ordering property. A superseded run's completion must not reset the
    /// state its replacement now owns — otherwise taking over would leave the
    /// new message playing while the button claimed nothing was.
    @Test
    func aSupersededRunCannotResetTheNewOwnersState() async {
        await toggle(messageA)
        // Grab A's completion before B takes over, then take over and fire it.
        let staleFinish = player.pendingFinish
        await toggle(messageB)
        #expect(viewModel.state == .speaking(messageID: messageB))

        staleFinish?()

        #expect(viewModel.state == .speaking(messageID: messageB))
        #expect(viewModel.isActive(messageB))
    }

    /// Taking over while the FIRST message is still synthesizing has to cancel
    /// that synthesis, or its result would arrive later and start playing over
    /// the message the user actually chose.
    @Test
    func takeoverDuringSynthesisCancelsTheAbandonedRun() async {
        let gate = AsyncGate()
        service.synthesizeGates = [gate]

        let first = Task { await toggle(messageA) }
        while gate.waiterCount == 0 { await Task.yield() }
        #expect(viewModel.state == .synthesizing(messageID: messageA))

        // B takes over while A is parked mid-synthesis.
        await toggle(messageB)
        gate.open()
        await first.value

        // A never reached the player; B is the only thing that played.
        #expect(player.playCount == 1)
        #expect(viewModel.state == .speaking(messageID: messageB))
    }

    @Test
    func stopWhileSynthesizingPlaysNothing() async {
        let gate = AsyncGate()
        service.synthesizeGates = [gate]

        let running = Task { await toggle(messageA) }
        while gate.waiterCount == 0 { await Task.yield() }

        viewModel.stop()
        gate.open()
        await running.value

        #expect(player.playCount == 0)
        #expect(viewModel.state == .idle)
    }

    // MARK: - Refusals that never reach the engine

    @Test
    func nilModelIDIsReportedWithoutCallingTheEngine() async {
        await toggle(messageA, modelID: nil)

        #expect(service.synthesizeCallCount == 0)
        #expect(player.playCount == 0)
        #expect(
            viewModel.state
                == .failed(
                    messageID: messageA, message: SpeechPlaybackViewModel.noModelMessage))
    }

    @Test
    func emptyTextIsReportedWithoutCallingTheEngine() async {
        await toggle(messageA, text: "   \n ")

        #expect(service.synthesizeCallCount == 0)
        #expect(player.playCount == 0)
        #expect(
            viewModel.state
                == .failed(
                    messageID: messageA, message: SpeechPlaybackViewModel.emptyTextMessage))
    }

    /// A sample rate the RIFF header cannot describe makes `WAVEncoder.encode`
    /// return nil. That is a real failure the user should see, not a button
    /// that quietly does nothing.
    @Test
    func unencodableAudioIsReportedAndNeverPlayed() async {
        service.speech = AudioEngine.Speech(samples: [0.1, 0.2], sampleRate: 0)

        await toggle(messageA)

        #expect(player.playCount == 0)
        #expect(
            viewModel.state
                == .failed(
                    messageID: messageA, message: SpeechPlaybackViewModel.unencodableMessage))
    }

    // MARK: - Failures

    @Test
    func synthesisFailureIsReportedAndNothingPlays() async {
        service.nextSynthesizeFails = true

        await toggle(messageA)

        #expect(player.playCount == 0)
        #expect(viewModel.errorMessage(for: messageA) != nil)
        #expect(viewModel.isActive(messageA) == false)
    }

    @Test
    func playbackFailureIsReported() async {
        player.nextPlayFails = true

        await toggle(messageA)

        #expect(service.synthesizeCallCount == 1)
        #expect(viewModel.errorMessage(for: messageA) != nil)
        #expect(viewModel.isActive(messageA) == false)
    }

    /// An error belongs to the message that produced it. Reporting it on every
    /// bubble would put a warning under replies that were never asked to speak.
    @Test
    func errorsAreScopedToTheFailingMessage() async {
        service.nextSynthesizeFails = true

        await toggle(messageA)

        #expect(viewModel.errorMessage(for: messageA) != nil)
        #expect(viewModel.errorMessage(for: messageB) == nil)
    }

    /// Clicking again after a failure retries rather than being wedged in the
    /// error state.
    @Test
    func retryAfterFailureWorks() async {
        service.nextSynthesizeFails = true
        await toggle(messageA)
        #expect(viewModel.errorMessage(for: messageA) != nil)

        await toggle(messageA)

        #expect(viewModel.state == .speaking(messageID: messageA))
        #expect(player.playCount == 1)
    }
}
