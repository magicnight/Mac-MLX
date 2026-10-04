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

    /// How many callers joined a fetch that was already running. A test hook:
    /// it is how a test knows a second caller is parked on the first fetch
    /// before it lets that fetch finish.
    private(set) var joins = 0

    init() {}

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
            joins += 1
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
