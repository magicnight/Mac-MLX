// Copyright © 2026 macMLX. English comments only.

/// A stable ``BatchGenerationServing`` seam that forwards to whatever batch-capable
/// engine is resident RIGHT NOW, resolved fresh on every call through a provider
/// closure — the batched-path analogue of ``HummingbirdServer``'s `engineProvider`
/// (SRV-1).
///
/// ## Why a provider, not a fixed engine
/// The GUI's `ModelPool` mints a NEW `MLXSwiftEngine` per model, so a seam holding
/// one fixed engine would keep draining/submitting against the engine that was
/// resident when the server started, not the one actually loaded now. Re-resolving
/// per call keeps `submit`/`drainForModelChange` pointed at the live engine — and,
/// crucially, makes `drainForModelChange` drain the OLD engine's cohort BEFORE a
/// swap (the provider still resolves the outgoing engine at drain time, since the
/// swap has not happened yet).
///
/// The provider returns `nil` (→ `submit` returns `nil`, `drainForModelChange` is a
/// no-op) whenever no engine is resident or the resident engine does not support
/// batching, so a non-batch engine transparently keeps the legacy single-stream
/// path. The CLI, whose engine reference is fixed, can pass its engine directly
/// instead of wrapping it here.
public final class ProvidedBatchServing: BatchGenerationServing {
    private let provider: @Sendable () async -> (any BatchGenerationServing)?

    /// - Parameter provider: resolves the currently-resident batch-capable seam
    ///   (typically `await pool.activeEngine as? any BatchGenerationServing`), or
    ///   `nil` when none is resident.
    public init(provider: @escaping @Sendable () async -> (any BatchGenerationServing)?) {
        self.provider = provider
    }

    public func submit(
        _ request: GenerateRequest
    ) async -> AsyncThrowingStream<GenerateChunk, Error>? {
        await provider()?.submit(request)
    }

    public func drainForModelChange() async {
        await provider()?.drainForModelChange()
    }
}
