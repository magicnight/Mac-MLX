// Copyright © 2026 macMLX. English comments only.

import Foundation
import MLXLMCommon

/// Off-actor serializer for the cold tier's safetensors writes (Wave 3 Stage 3a).
///
/// `savePromptCache` serialises a KV snapshot synchronously — measured ~43 ms for
/// a 300 MiB snapshot, scaling with cache size. Run on ``PromptCacheStore``'s own
/// executor (as Stage 2b did) that call stalls EVERY concurrent
/// `fetchNearest`/`insert` for the whole write. This actor moves the blocking IO
/// onto ITS executor instead: a `PromptCacheStore` demote spawns a task that
/// `await`s ``write(snapshot:to:metadata:)``, so the store's executor is free the
/// instant it suspends at that await. Serial by actor isolation — one write runs
/// at a time — which bounds peak IO and, combined with the store's
/// `maxInFlightWrites` gate, peak retained-snapshot memory.
///
/// **Atomicity.** `savePromptCache` writes in place (`O_TRUNC`, non-atomic), so a
/// crash or concurrent read mid-write could observe a torn file. Content-address
/// + parse-validation already demote a torn file to a load-miss (never wrong
/// output), but Stage 3a still writes to a temp sibling and atomically `rename`s
/// it into place so no reader EVER sees a partial file at the canonical path.
///
/// **Cancellation.** ``PromptCacheStore/clearAll()`` cancels every in-flight write
/// before wiping the cold root. ``write(snapshot:to:metadata:)`` checks
/// cancellation immediately before the rename, so a cancelled write aborts with
/// its temp file cleaned up and NOTHING lands at the canonical path — no entry can
/// resurrect past the wipe (invariant I4).
///
/// **The Sendable boundary.** The snapshot crosses from the store's isolation
/// domain into this actor's via ``PromptCacheSnapshot`` — the module's existing
/// `@unchecked Sendable` KV carrier, whose documented invariant is exclusive
/// ownership of its caches. That invariant holds exactly here: the store only ever
/// demotes caches it has just `pop`-ed out of the hot trie (``evictOne``), which
/// nothing else references, so handing them to this writer transfers sole
/// ownership with no aliasing and no copy. (A raw `sending [any KVCache]` transfer
/// cannot be expressed: a value stored in the actor-isolated trie is permanently
/// region-merged with the actor, so the compiler rejects every attempt to extract
/// it as `sending`. The snapshot carrier is the one honest handoff.)
actor ColdTierWriter {

    /// Serialise `snapshot` to a temp sibling of `finalURL`, then atomically
    /// rename it into place. Serial by actor isolation — `savePromptCache` blocks
    /// THIS actor's executor (intended), never the store's. Cancellation is
    /// checked right before the rename so a `clearAll`-cancelled write aborts
    /// before any file lands, its temp cleaned up.
    func write(
        snapshot: PromptCacheSnapshot, to finalURL: URL, metadata: [String: String]
    ) async throws {
        Self.ensureParentDirectory(of: finalURL)
        let tmp = Self.temporaryURL(for: finalURL)
        do {
            try savePromptCache(url: tmp, cache: snapshot.caches, metadata: metadata)
            // Abort a `clearAll`-cancelled write before it can land at the
            // canonical path (I4). Done AFTER the temp write so the expensive
            // serialisation isn't wasted needlessly, and BEFORE the rename so a
            // cancelled write never becomes visible.
            try Task.checkCancellation()
            try Self.atomicallyReplace(finalURL, with: tmp)
        } catch {
            // Any failure — serialisation error, cancellation, rename error —
            // leaves no torn canonical file: best-effort remove the temp, rethrow
            // so the store's `finishWrite` reconciles the phantom index record.
            try? FileManager.default.removeItem(at: tmp)
            throw error
        }
    }

    /// Synchronous atomic write for the store's backpressure fallback: when too
    /// many detached writes are already in flight, the store degrades to writing
    /// inline on its own executor (bounded memory beats unbounded background
    /// writers). Same temp-then-rename atomicity as ``write(snapshot:to:metadata:)``
    /// minus the cancellation check — the caller is a foreground `demoteToCold`
    /// waiting on the result, not a cancellable background task.
    nonisolated static func writeSynchronously(
        snapshot: PromptCacheSnapshot, to finalURL: URL, metadata: [String: String]
    ) throws {
        ensureParentDirectory(of: finalURL)
        let tmp = temporaryURL(for: finalURL)
        do {
            try savePromptCache(url: tmp, cache: snapshot.caches, metadata: metadata)
            try atomicallyReplace(finalURL, with: tmp)
        } catch {
            try? FileManager.default.removeItem(at: tmp)
            throw error
        }
    }

    // MARK: - Shared filesystem helpers (nonisolated: pure path work)

    /// Create the sharded parent directory if missing. Defensive against
    /// ``PromptCacheStore/clearAll()`` having wiped the cold root out from under an
    /// in-flight write between its spawn and its execution.
    nonisolated static func ensureParentDirectory(of finalURL: URL) {
        try? FileManager.default.createDirectory(
            at: finalURL.deletingLastPathComponent(), withIntermediateDirectories: true)
    }

    /// A unique temp sibling of `finalURL` in the SAME directory (hence same
    /// volume, so the rename is atomic). Two constraints shape the name:
    /// it must KEEP the `.safetensors` extension — MLX's `save` rejects any other
    /// (`LoadSaveError.unknownExtension`) — yet must be invisible to
    /// ``PromptCacheStore/pruneColdDirectory``, whose byte-budget scan counts
    /// `*.safetensors` files. The leading-dot (hidden) prefix threads both: the
    /// extension is still `safetensors`, but the prune enumerator's
    /// `.skipsHiddenFiles` skips it, so a mid-write temp is never counted toward
    /// (nor evicted by) the cap.
    nonisolated static func temporaryURL(for finalURL: URL) -> URL {
        finalURL.deletingLastPathComponent()
            .appending(
                path: ".tmp-\(UUID().uuidString).safetensors",
                directoryHint: .notDirectory)
    }

    /// Best-effort startup cleanup for write temporaries orphaned by a hard kill
    /// between ``write(snapshot:to:metadata:)``'s temp `savePromptCache` and its
    /// atomic rename (Wave 3 Stage 3a).
    ///
    /// A crash in that window leaves a hidden `.tmp-<uuid>.safetensors` sibling
    /// behind. Because it is hidden by design (see ``temporaryURL(for:)``),
    /// ``PromptCacheStore/pruneColdDirectory`` — whose enumerator passes
    /// `.skipsHiddenFiles` so it never counts or evicts a live temp mid-write —
    /// will ALSO never reclaim a dead one: left unswept, each crash permanently
    /// leaks one full snapshot's worth of disk, unbounded across restarts.
    ///
    /// `PromptCacheStore.init` calls this once, right after creating the cold
    /// root and BEFORE the byte-cap prune / manifest rebuild. That ordering is
    /// safe specifically because it's startup: no write can possibly be in
    /// flight yet (this store instance hasn't spawned one), so every matching
    /// temp found here is unambiguously a crash orphan, never a live write's
    /// temp snatched out from under it. `nonisolated static` (no actor state
    /// needed) so `init` can call it before the actor is fully initialised,
    /// mirroring ``PromptCacheStore/pruneColdDirectory``'s shape.
    nonisolated static func sweepStaleColdTemporaries(root: URL) {
        let fm = FileManager.default
        guard
            let enumerator = fm.enumerator(
                at: root,
                includingPropertiesForKeys: [.isRegularFileKey],
                // Deliberately WITHOUT `.skipsHiddenFiles`: the files this sweep
                // exists to find are exactly the hidden ones the cap-prune skips.
                options: [],
                errorHandler: { _, _ in true }  // skip unreadable subtrees, keep going
            )
        else { return }
        for case let url as URL in enumerator {
            let name = url.lastPathComponent
            guard name.hasPrefix(".tmp-"), url.pathExtension == "safetensors" else { continue }
            try? fm.removeItem(at: url)  // best-effort, matching the tier's leniency elsewhere
        }
    }

    /// Atomically move `tmp` onto `finalURL`, overwriting any existing file.
    /// POSIX `rename(2)` is atomic on a single volume and — unlike
    /// `FileManager.moveItem` — overwrites an existing destination, which the
    /// content-addressed cold tier needs when the same snapshot is re-demoted onto
    /// an already-present file.
    nonisolated static func atomicallyReplace(_ finalURL: URL, with tmp: URL) throws {
        var status: Int32 = 0
        var savedErrno: Int32 = 0
        tmp.path.withCString { tmpPath in
            finalURL.path.withCString { finalPath in
                status = rename(tmpPath, finalPath)
                savedErrno = errno
            }
        }
        if status != 0 { throw ColdTierWriteError.renameFailed(code: savedErrno) }
    }
}

/// A failure of the cold-tier atomic write's final `rename` step, carrying the
/// POSIX `errno` for diagnostics. Any throw here surfaces to the store's
/// `finishWrite`, which reconciles the phantom index record.
enum ColdTierWriteError: Error, Equatable {
    case renameFailed(code: Int32)
}
