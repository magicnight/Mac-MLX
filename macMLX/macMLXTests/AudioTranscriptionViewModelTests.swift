// AudioTranscriptionViewModelTests.swift
// macMLXTests
//
// State-machine invariants for the chat composer's audio attachment.
//
// The property under test throughout is that the composer is written to on
// exactly one path — a successful transcription with real text — and that every
// other outcome leaves it alone AND says why. A button that sometimes does
// nothing without explaining is the specific failure these guard against.

import Foundation
import Testing
import MacMLXCore
@testable import macMLX

@Suite("AudioTranscriptionViewModel")
@MainActor
final class AudioTranscriptionViewModelTests {

    private let service = StubAudioService()
    private let viewModel: AudioTranscriptionViewModel
    private let audioURL = URL(fileURLWithPath: "/nonexistent/recording.wav")

    /// Text handed to the composer, in order. Empty means the composer was
    /// never touched — the assertion most of these tests turn on.
    private var delivered: [String] = []

    init() {
        viewModel = AudioTranscriptionViewModel(service: service)
    }

    private func transcribe(modelID: String? = "openai/whisper-tiny") async {
        await viewModel.transcribe(audioURL: audioURL, modelID: modelID) { [weak self] text in
            self?.delivered.append(text)
        }
    }

    // MARK: - Success

    @Test
    func successfulTranscriptionDeliversTextAndSettlesIdle() async {
        service.transcript = "hello there"

        await transcribe()

        #expect(delivered == ["hello there"])
        #expect(viewModel.state == .idle)
        #expect(viewModel.errorMessage == nil)
    }

    /// Models in this family routinely emit a leading space or trailing
    /// newline. Delivering that verbatim would put stray whitespace in the
    /// composer on every use.
    @Test
    func deliveredTextIsTrimmed() async {
        service.transcript = "  hello there\n"

        await transcribe()

        #expect(delivered == ["hello there"])
    }

    @Test
    func theResolvedModelIDIsWhatReachesTheEngine() async {
        await transcribe(modelID: "nvidia/parakeet-tdt")

        #expect(service.transcribeCalls.count == 1)
        #expect(service.transcribeCalls.first?.modelID == "nvidia/parakeet-tdt")
        #expect(service.transcribeCalls.first?.audioURL == audioURL)
    }

    // MARK: - Refusals that never reach the engine

    /// An unresolvable model must be reported, not guessed around — and must
    /// not cost a load attempt.
    @Test
    func nilModelIDIsReportedWithoutCallingTheEngine() async {
        await transcribe(modelID: nil)

        #expect(service.transcribeCallCount == 0)
        #expect(delivered.isEmpty)
        #expect(viewModel.state == .failed(message: AudioTranscriptionViewModel.noModelMessage))
        #expect(viewModel.errorMessage == AudioTranscriptionViewModel.noModelMessage)
    }

    // MARK: - Failures

    @Test
    func engineFailureIsSurfacedAndComposerUntouched() async {
        service.nextTranscribeFails = true

        await transcribe()

        #expect(delivered.isEmpty)
        #expect(viewModel.errorMessage != nil)
        #expect(viewModel.state != .idle)
    }

    /// An empty transcript is a real outcome — silence, or audio with no
    /// speech. Appending nothing and calling it success would be
    /// indistinguishable from a broken button, so it is reported.
    @Test
    func emptyTranscriptIsReportedRatherThanDeliveredSilently() async {
        service.transcript = ""

        await transcribe()

        #expect(delivered.isEmpty)
        #expect(viewModel.state == .failed(message: AudioTranscriptionViewModel.noSpeechMessage))
    }

    @Test
    func whitespaceOnlyTranscriptIsReportedTheSameWay() async {
        service.transcript = "   \n  "

        await transcribe()

        #expect(delivered.isEmpty)
        #expect(viewModel.state == .failed(message: AudioTranscriptionViewModel.noSpeechMessage))
    }

    @Test
    func clearErrorReturnsToIdle() async {
        service.nextTranscribeFails = true
        await transcribe()
        #expect(viewModel.errorMessage != nil)

        viewModel.clearError()

        #expect(viewModel.state == .idle)
        #expect(viewModel.errorMessage == nil)
    }

    // MARK: - In-flight state and cancellation

    /// While the model works, the composer has to show something. The gate
    /// makes "is it transcribing right now" an exact observation instead of a
    /// sleep-and-hope.
    @Test
    func stateIsTranscribingWhileTheEngineWorks() async {
        let gate = AsyncGate()
        service.transcribeGates = [gate]

        let running = Task { await transcribe() }
        while gate.waiterCount == 0 { await Task.yield() }

        #expect(viewModel.isTranscribing)
        #expect(viewModel.state == .transcribing)

        gate.open()
        await running.value

        #expect(viewModel.isTranscribing == false)
    }

    /// Cancel is a user action, not a failure: the composer stays untouched
    /// and NO error is shown — the user already knows what happened.
    @Test
    func cancelDeliversNothingAndShowsNoError() async {
        let gate = AsyncGate()
        service.transcribeGates = [gate]

        let running = Task { await transcribe() }
        while gate.waiterCount == 0 { await Task.yield() }

        viewModel.cancel()
        gate.open()
        await running.value

        #expect(delivered.isEmpty)
        #expect(viewModel.state == .idle)
        #expect(viewModel.errorMessage == nil)
    }

    /// Picking a second file supersedes the first: the older transcript is
    /// stale by definition and must not land in the composer after the newer
    /// one — otherwise the two arrive in completion order, not pick order.
    @Test
    func aSecondPickSupersedesTheFirstResult() async {
        let firstGate = AsyncGate()
        service.transcribeGates = [firstGate]
        service.transcript = "stale result"

        let first = Task { await transcribe() }
        while firstGate.waiterCount == 0 { await Task.yield() }

        // Second pick, ungated, completes while the first is still parked.
        service.transcript = "fresh result"
        await transcribe()

        firstGate.open()
        await first.value

        #expect(delivered == ["fresh result"])
    }
}
