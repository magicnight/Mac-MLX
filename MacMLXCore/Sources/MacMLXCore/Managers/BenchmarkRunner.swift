// BenchmarkRunner.swift
// MacMLXCore
//
// Runs a canonical benchmark workload (issue #22): one warm-up iteration
// followed by N measured generations, then aggregates the medians.
//
// Design split: the runner doesn't own the engine or do model loading.
// The caller (BenchmarkViewModel in the app) loads/unloads and passes a
// `generate(_:)` closure in. Keeps the runner pure — testable with a
// canned-stream mock, no MLX dependency.

import Foundation

// MARK: - Runner

/// Runs a timed generation benchmark against a provided `generate`
/// closure and aggregates the medians across multiple iterations.
///
/// Not tied to any particular engine: the caller supplies the generate
/// callback, so unit tests can hand in a canned `AsyncThrowingStream` of
/// `GenerateChunk` values without standing up MLX.
public actor BenchmarkRunner {

    /// `@Sendable` callback that produces a streaming response.
    public typealias GenerateCallback =
        @Sendable (GenerateRequest) -> AsyncThrowingStream<GenerateChunk, Error>

    private let generate: GenerateCallback
    /// Token → character ratio used when synthesising the canonical
    /// prompt. `4.0` is a reasonable English average; the exact value
    /// doesn't matter because the model's tokenizer reports the real
    /// count in `usage.promptTokens`.
    private let charsPerTokenHeuristic: Double = 4.0

    public init(generate: @escaping GenerateCallback) {
        self.generate = generate
    }

    // MARK: - Public API

    /// Run the benchmark workload.
    ///
    /// - Parameters:
    ///   - modelID: label used both in the `GenerateRequest.model` and in
    ///     the resulting `BenchmarkResult.modelID`.
    ///   - engineID / engineVersion: provenance for the result row.
    ///   - promptTokens: approximate target length of the synthetic prompt.
    ///     The runner generates a long filler string and relies on the
    ///     engine's tokenizer to report the actual `usage.promptTokens`.
    ///   - generationTokens: upper bound on generated tokens per iteration
    ///     (passed as `maxTokens` on the request).
    ///   - runs: number of **measured** iterations. Must be ≥ 1. A single
    ///     un-counted warm-up always runs first.
    ///   - modelLoadTimeS: caller-measured cold-load time, passed through
    ///     unchanged into the result. Pass 0 when the model was already
    ///     loaded.
    ///   - notes: user-supplied free-form notes.
    public func run(
        modelID: String,
        engineID: EngineID,
        engineVersion: String,
        promptTokens: Int = 512,
        generationTokens: Int = 200,
        runs: Int = 3,
        modelLoadTimeS: Double = 0,
        notes: String = ""
    ) async throws -> BenchmarkResult {
        precondition(runs > 0, "runs must be >= 1")

        // Warm-up (not counted — covers CPU-frequency ramp, JIT compilation,
        // initial buffer allocation, first-token kernel launches).
        _ = try await measureOne(
            modelID: modelID,
            promptTokens: 64,
            generationTokens: 20
        )

        // Measured iterations.
        var samples: [SingleSample] = []
        samples.reserveCapacity(runs)
        for _ in 0..<runs {
            let s = try await measureOne(
                modelID: modelID,
                promptTokens: promptTokens,
                generationTokens: generationTokens
            )
            samples.append(s)
        }

        return aggregate(
            samples: samples,
            modelID: modelID,
            engineID: engineID,
            engineVersion: engineVersion,
            promptTokens: promptTokens,
            runs: runs,
            modelLoadTimeS: modelLoadTimeS,
            notes: notes
        )
    }

    // MARK: - Single-iteration measurement

    private struct SingleSample {
        let promptTokens: Int
        let completionTokens: Int
        let ttftSeconds: Double         // wall-clock: request-send → first token
        let generationSeconds: Double   // first token → last token
        let peakResidentBytes: UInt64
    }

    private func measureOne(
        modelID: String,
        promptTokens: Int,
        generationTokens: Int
    ) async throws -> SingleSample {
        let request = makeRequest(
            modelID: modelID,
            promptTokens: promptTokens,
            generationTokens: generationTokens
        )

        // Start a peak-memory sampler that polls every 50ms until we
        // cancel it at end-of-stream. Captures the highest RSS we see.
        let memoryProbe = PeakMemorySampler.start()

        let start = Date()
        var firstTokenTime: Date?
        var endTime: Date?
        var promptCount: Int = promptTokens
        var completionCount: Int = 0

        let stream = generate(request)
        for try await chunk in stream {
            if firstTokenTime == nil && !chunk.text.isEmpty {
                firstTokenTime = Date()
            }
            if let usage = chunk.usage {
                // Prefer the engine's authoritative counts if it ever
                // emits them (most do only on the terminal chunk).
                if usage.promptTokens > 0 { promptCount = usage.promptTokens }
                if usage.completionTokens > 0 { completionCount = usage.completionTokens }
            }
            if chunk.finishReason != nil {
                endTime = Date()
            }
        }
        endTime = endTime ?? Date()
        let peak = await memoryProbe.stopAndCollect()

        let ft = firstTokenTime ?? endTime!  // degenerate: no tokens
        return SingleSample(
            promptTokens: promptCount,
            completionTokens: completionCount,
            ttftSeconds: ft.timeIntervalSince(start),
            generationSeconds: max(0.001, endTime!.timeIntervalSince(ft)),
            peakResidentBytes: peak
        )
    }

    // MARK: - Prompt synthesis

    private func makeRequest(
        modelID: String,
        promptTokens: Int,
        generationTokens: Int
    ) -> GenerateRequest {
        let targetChars = max(8, Int(Double(promptTokens) * charsPerTokenHeuristic))
        let prompt = Self.fillerPrompt(minChars: targetChars)
        return GenerateRequest(
            model: modelID,
            messages: [ChatMessage(role: .user, content: prompt)],
            systemPrompt: nil,
            parameters: GenerationParameters(
                temperature: 0.7,
                topP: 0.95,
                maxTokens: generationTokens,
                stream: true
            )
        )
    }

    /// Lorem-Ipsum-esque filler that drives the tokenizer to a target length.
    /// Deterministic so two runs of the same size benchmark the same way.
    private static func fillerPrompt(minChars: Int) -> String {
        let base = """
        Write a concise technical summary of the following topic. Cover \
        the key background, the recent developments, and the practical \
        implications. Keep the tone neutral and the length reasonable. \
        Topic: the evolution of tensor-level operator fusion in modern \
        machine-learning compilers, focusing on how schedule primitives, \
        autotuning, and hardware-aware kernel selection changed over the \
        last five years on Apple Silicon and on consumer GPUs. \

        """
        var s = "Please produce the following article. "
        while s.count < minChars {
            s += base
        }
        return s
    }

    // MARK: - Aggregation

    private func aggregate(
        samples: [SingleSample],
        modelID: String,
        engineID: EngineID,
        engineVersion: String,
        promptTokens: Int,
        runs: Int,
        modelLoadTimeS: Double,
        notes: String
    ) -> BenchmarkResult {
        // Use actual engine-reported token counts if any sample has them,
        // otherwise fall back to the requested numbers.
        let promptTokensFinal = samples.last(where: { $0.promptTokens > 0 })?.promptTokens ?? promptTokens
        let completionTokensFinal = samples
            .compactMap { $0.completionTokens > 0 ? $0.completionTokens : nil }
            .reduce(0, +) / max(1, samples.count)

        let promptTPS = Self.median(
            samples.map { Double($0.promptTokens) / max(0.001, $0.ttftSeconds) }
        )
        let generationTPS = Self.median(
            samples.map { Double($0.completionTokens) / $0.generationSeconds }
        )
        let ttftMs = Self.median(samples.map { $0.ttftSeconds * 1000.0 })
        let peakBytes = samples.map(\.peakResidentBytes).max() ?? 0
        let peakGB = Double(peakBytes) / 1_073_741_824.0

        return BenchmarkResult(
            modelID: modelID,
            engineID: engineID,
            promptTokens: promptTokensFinal,
            completionTokens: completionTokensFinal,
            runs: runs,
            promptTPS: promptTPS,
            generationTPS: generationTPS,
            ttftMs: ttftMs,
            memoryUsedGB: peakGB,
            modelLoadTimeS: modelLoadTimeS,
            system: HardwareInfo.snapshot(),
            macMLXVersion: Self.bundleShortVersion(),
            engineVersion: engineVersion,
            notes: notes
        )
    }

    // MARK: - Math helpers

    /// Median of a sorted-then-picked-middle series. For 1 sample returns
    /// it verbatim; for 2 samples returns the mean; for 3+ the true middle
    /// (odd) or mean of the two middles (even). Never crashes on empty —
    /// returns 0.
    internal static func median(_ values: [Double]) -> Double {
        guard !values.isEmpty else { return 0 }
        let sorted = values.sorted()
        let n = sorted.count
        if n.isMultiple(of: 2) {
            return (sorted[n/2 - 1] + sorted[n/2]) / 2
        }
        return sorted[n/2]
    }

    private static func bundleShortVersion() -> String {
        Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String ?? ""
    }
}

// MARK: - Memory probe

/// Polling-based peak-RSS sampler. Runs a detached task that reads
/// `mach_task_basic_info.resident_size` every 50ms and keeps a running
/// max, until `stopAndCollect()` is called.
///
/// 50ms is fine-grained enough to catch short-lived spikes on a typical
/// few-second benchmark run, and cheap enough not to perturb the
/// measurement. `task_info` is a non-blocking read against the current
/// process.
private actor PeakMemorySampler {
    private var peak: UInt64 = 0
    private var sampleTask: Task<Void, Never>?

    static func start() -> PeakMemorySampler {
        let probe = PeakMemorySampler()
        // Detached so the sampler survives the actor hop into the
        // benchmark runner's continuation. The task is retained on the
        // actor so `stopAndCollect()` can cancel + await deterministic
        // teardown — pre-v0.3 we relied on a plain `active` flag which
        // let the loop run for up to ~50ms after stop returned and held
        // `self` until the next tick. Reviewer-flagged HIGH.
        let task = Task.detached(priority: .utility) { [weak probe] in
            while !Task.isCancelled {
                await probe?.recordSample()
                try? await Task.sleep(nanoseconds: 50_000_000)  // 50ms
            }
        }
        Task { await probe.setTask(task) }
        return probe
    }

    private func setTask(_ task: Task<Void, Never>) { self.sampleTask = task }

    fileprivate func recordSample() {
        peak = max(peak, MemoryProbe.residentMemoryBytes())
    }

    func stopAndCollect() async -> UInt64 {
        sampleTask?.cancel()
        // Await the sampling task's completion so we don't race with a
        // final sample landing after this method returns.
        _ = await sampleTask?.value
        sampleTask = nil
        // One final read so a brief generation at least returns a number.
        peak = max(peak, MemoryProbe.residentMemoryBytes())
        return peak
    }
}
