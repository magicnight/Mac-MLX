import Foundation

/// One download per repo at a time, shared process-wide.
///
/// Upstream's resolve-or-download treats a model directory without a complete
/// `.safetensors` as stale and clears it, so two concurrent downloads of one
/// repo delete each other's files. Every `AudioEngine` in the process routes
/// its fetches through ``shared`` — the server's engine and the app's share
/// one cache directory — so a second request for a model that is still
/// downloading joins the download instead of starting another. Tests use a
/// private instance so their fake fetches cannot meet.
actor AudioSnapshotFetcher {
    static let shared = AudioSnapshotFetcher()

    private var inFlight: [String: Task<Void, any Error>] = [:]

    /// Called with the repo id each time a caller joins a fetch that was
    /// already running, just before that caller starts waiting on it. A test
    /// hook: it is how a test learns that a second caller has found the
    /// first fetch and is about to wait on it, before the test lets that
    /// fetch finish. Reported as an event rather than kept as a count, so
    /// the test can wait for it instead of polling against a deadline.
    private let onJoin: @Sendable (_ modelID: String) -> Void

    init(onJoin: @escaping @Sendable (_ modelID: String) -> Void = { _ in }) {
        self.onJoin = onJoin
    }

    /// Run `fetch` for `modelID`, or wait for the one already running.
    ///
    /// An `EngineError` thrown by the fetch passes through; anything else
    /// becomes ``EngineError/modelLoadFailed(reason:)``, which the server
    /// reports as 500 `load_failed`. The download itself is never cancelled:
    /// whether it started the download or joined it, a caller whose task is
    /// cancelled waits the download out, and a caller that arrives in the
    /// moment between the download finishing and its starter clearing the
    /// entry gets the finished result, a failure included.
    func fetch(_ modelID: String, using fetch: @escaping AudioEngine.SnapshotFetch) async throws {
        if let running = inFlight[modelID] {
            onJoin(modelID)
            do { try await running.value } catch { throw Self.loadFailure(error) }
            return
        }
        let task = Task { try await fetch(modelID) }
        inFlight[modelID] = task
        defer { inFlight[modelID] = nil }
        do { try await task.value } catch { throw Self.loadFailure(error) }
    }

    private static func loadFailure(_ error: any Error) -> EngineError {
        (error as? EngineError) ?? .modelLoadFailed(reason: error.localizedDescription)
    }
}
