// Copyright © 2026 macMLX. English comments only.

/// A2d server↔scheduler seam: the boundary ``HummingbirdServer`` calls to run a
/// batchable request on the continuous-batching path (``BatchScheduler``) instead
/// of the legacy single-stream engine path.
///
/// Injected into the server like `engineProvider` / `loadHook` / `inFlightHook`
/// — a closure-style dependency — so `MacMLXCore`'s server stays free of the
/// scheduler-construction concern and can be exercised with a stub. This mirrors
/// the MLX-free seam philosophy A2c already uses for ``BatchInferenceCore`` (one
/// production implementation + scripted stubs in tests).
///
/// ## Why a seam rather than the scheduler directly
/// A production model is only ever resident inside a `ModelContainer`, which (by
/// design and by Swift 6's `perform<R: Sendable>` constraint) will not surrender
/// its non-`Sendable` `model`/`tokenizer` to construct a separate
/// ``BatchScheduler`` actor. Building + owning the scheduler, and coordinating its
/// MLX work against every other `container.perform` for the same model, therefore
/// belongs to the layer that owns the container's lifecycle — reached through
/// this seam, not baked into the server. See the A2d hand-off notes for the
/// construction wave.
///
/// ## Contract
/// - ``submit(_:)`` returns a per-request `AsyncThrowingStream<GenerateChunk,
///   Error>` — the SAME shape `InferenceEngine.generate` returns, so the server's
///   `ChunkIteratorBox` + stall-watchdog + SSE machinery is reused unchanged —
///   when the RESIDENT model can batch the request. It returns `nil` to signal
///   "not batchable — use the legacy single-stream path" (a VLM tower, an
///   architecture off the verified-`ropeOffset` allowlist, a non-dense cache, or
///   a request naming a model that is not the resident one). Returning `nil`
///   performs NO generation work, so the caller falls back with no double-billing
///   and no double in-flight accounting. See "Admission is a promise" below for
///   what a non-nil return commits an implementation to.
/// - ``drainForModelChange()`` quiesces before a cold-swap/unload of the resident
///   model — the SRV-2 swap-under-lock invariant generalized to drain-after-swap:
///   stop admitting new rows and drain the active cohort to zero (cancelling live
///   rows if it must bound the wait) so the model is never swapped out from under
///   live rows. It MUST NOT deadlock: a swap waits for rows, and rows never wait
///   for a swap (the batched path never holds the server's FIFO generation lock).
///   See "The submit/drain epoch" and "Drain must never re-enter the caller's
///   lock" below for what this requires of an implementation.
///
/// ## The submit/drain epoch (HIGH-1)
/// `HummingbirdServer` calls ``submit(_:)`` WITHOUT holding its FIFO generation
/// lock — bypassing that lock is the entire point of the batched path — while
/// ``drainForModelChange()`` is always called FROM UNDER that lock (via
/// `ensureModelLoaded`, reached from `beginGeneration`, `handleLoadModel`, and
/// `handleUnloadModel`). Consequently a `submit` call for one request and a
/// `drainForModelChange` call triggered by another request CAN run concurrently
/// — the server provides no mutual exclusion between them, and by design it
/// cannot: the very lock a drain runs under is the lock the batched path exists
/// to bypass. An implementation of this protocol MUST supply the missing mutual
/// exclusion itself:
/// - From the moment a `drainForModelChange` call begins until the model swap or
///   unload it is guarding has FINISHED, every `submit` call — concurrent with,
///   or arriving after, the start of that drain but before the swap completes —
///   MUST return `nil`. Those requests are routed to the legacy single-stream
///   path instead of racing the swap; there is no partial admission.
/// - `drainForModelChange`, the admission-gating that begins with it, and the
///   swap/unload it precedes together form ONE serial epoch on the actor that
///   owns admission (the scheduler actor). No row may be admitted partway
///   through a drain; no drain may begin partway through admitting a row.
/// - This is a MUST, not a "nice to have," precisely because the server has no
///   mechanism left to provide this guarantee itself — the batched path is BY
///   DESIGN lock-free with respect to `HummingbirdServer`'s FIFO generation lock
///   (that lock-freedom is what makes concurrent batched admission useful at
///   all), so the seam's own actor isolation is the ONLY place this invariant
///   can live.
///
/// ## A dropped stream must evict its row (HIGH-2)
/// `HummingbirdServer` calls ``submit(_:)`` BEFORE the HTTP response-body
/// closure ever runs (see `handleChatCompletions`), so admission — and per
/// "`submit` is a billing/timing anchor" below, decoding — begins the instant
/// `submit` returns a non-nil stream, regardless of whether the caller ever
/// consumes it. If the HTTP body is never driven to completion (the client
/// disconnects before the body starts, a stalled connection is abandoned, the
/// serving `Task` is cancelled), the returned `AsyncThrowingStream`'s iterator is
/// dropped without ever reaching `.finished`/`.failure` — and unless the
/// implementation notices, the row it admitted becomes an ORPHAN: it keeps
/// occupying a cohort slot and consuming decode steps indefinitely, for a
/// response nobody is listening to.
///
/// The stream returned by `submit` MUST therefore attach an `onTermination`
/// handler (`AsyncThrowingStream<GenerateChunk, Error>.Continuation.onTermination`)
/// that evicts the corresponding row from the scheduler when the stream
/// terminates for ANY reason — INCLUDING `.cancelled`. Cancellation is the
/// COMMON case here, not a rare edge case: on the caller side, `ChunkIteratorBox`
/// is the ONLY holder of this stream's iterator, and its `deinit` is exactly
/// what fires `onTermination(.cancelled)` when a body that never ran drops the
/// box with nothing ever having consumed the stream. An implementation that only
/// evicts on a normal terminal case (`.finished` / `.failure`) and ignores
/// `.cancelled` WILL leak a row on every dropped connection.
///
/// ## Admission is a promise: no admit-then-fail (MEDIUM-7)
/// Every coverage check that decides whether a request CAN be served on the
/// batched path — VLM/multimodal-tower detection, the verified-`ropeOffset`
/// architecture allowlist, dense-vs-non-dense KV cache, and "does this request
/// name the currently-resident model" — MUST run to completion SYNCHRONOUSLY
/// inside the `submit` call, before it returns. Returning a non-nil stream is a
/// binding promise that this request will be served start-to-finish on the
/// batched path; there is NO post-admission escape hatch back to the legacy
/// path. Concretely: never return a stream first and only later discover
/// mid-generation that the request doesn't actually fit the resident cohort
/// (wrong cache layout, a VLM tower, an off-allowlist architecture, etc.) — all
/// of that is a pre-admission decision, made once, before `submit` returns.
///
/// ## Drain must never re-enter the caller's lock (open question, resolved)
/// ``drainForModelChange()`` — and everything it transitively calls — MUST NOT
/// attempt to acquire `HummingbirdServer`'s FIFO generation lock
/// (`acquireGenerationLock()` / `releaseGenerationLock()`). Every call site
/// (`ensureModelLoaded`, reached from `beginGeneration`, `handleLoadModel`,
/// `handleUnloadModel`) already HOLDS that lock for the full duration of the
/// `drainForModelChange()` call. Because the task awaiting `drainForModelChange`
/// IS the task holding the lock, a reentrant acquire attempt here is not a race
/// that might resolve favorably — it is a guaranteed, permanent self-deadlock:
/// no other task is left that could ever call `releaseGenerationLock()` to
/// unblock it.
///
/// ## `submit` is a billing/timing anchor, not a queueing op (open question, resolved)
/// A non-nil return from ``submit(_:)`` means admission — and therefore decoding
/// — has ALREADY started, the way continuous batching normally works: this seam
/// exposes no separate "enqueued but not yet running" state. Callers MUST treat
/// the row as in-flight (for accounting, in-flight/busy tracking, timeouts, or
/// anything else keyed on "is this request currently running") from the MOMENT
/// `submit` returns, not from whenever they get around to first consuming the
/// stream — anchoring on consumption-start would under-count how long the row
/// has actually been decoding, and marking after the fact leaves a window where
/// a concurrent eviction could race decode that has already begun.
/// `HummingbirdServer` marks its in-flight refcount (`markInFlight`, POOL-3)
/// immediately BEFORE calling `submit`, precisely so a concurrent LRU-driven
/// load can't steal the resident model out from under a row that — from the
/// seam's perspective — is already decoding by the time `submit` returns.
public protocol BatchGenerationServing: Sendable {
    /// Serve `request` on the continuous-batching path, or return `nil` to route
    /// it through the legacy single-stream path. Returning non-nil is a binding
    /// admission promise (see "Admission is a promise" on the type) and means
    /// decoding has already started (see "`submit` is a billing/timing anchor").
    /// The returned stream MUST evict its row on any termination, including a
    /// dropped/cancelled iterator (see "A dropped stream must evict its row").
    /// See the type's contract for full detail.
    func submit(_ request: GenerateRequest) async -> AsyncThrowingStream<GenerateChunk, Error>?

    /// Drain the active cohort before the resident model is swapped or unloaded.
    /// MUST NOT acquire the caller's generation lock (see "Drain must never
    /// re-enter the caller's lock") and MUST form a single serial epoch with
    /// admission-gating and the subsequent swap (see "The submit/drain epoch").
    /// See the type's contract for full detail.
    func drainForModelChange() async
}
