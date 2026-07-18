// Copyright © 2026 macMLX. English comments only.

import Foundation

/// The budget knobs a ``PromptCacheStore`` is built with, derived from the
/// user's persisted ``Settings`` and threaded to every construction site
/// (`MLXSwiftEngine`, the GUI `EngineCoordinator`, the CLI `CLIContext`).
///
/// Two-tier budget model:
///
/// - **Hot tier** — `hotBytes` is the PRIMARY in-RAM ceiling (fed straight into
///   `PromptCacheStore.maxBytes`, sourced from `Settings.kvCacheHotMB`).
///   `maxEntries` is a SECONDARY safety ceiling: for typical MB-scale KV
///   snapshots the byte budget is reached first and dominates, so `maxEntries`
///   only bites in a pathological many-tiny-entries case, bounding the trie /
///   LRU bookkeeping instead of letting it grow without limit.
///
/// - **Cold tier** — `coldEnabled` is the master opt-in (default `true`: the
///   cold tier already runs today, so enabling it *with* a byte budget is
///   strictly safer than the unbounded status quo). `coldCapBytes` caps the
///   on-disk directory; `PromptCacheStore.pruneCold` enforces it by mtime-LRU.
public struct PromptCacheConfig: Sendable, Equatable {

    /// Hot-tier byte ceiling (`PromptCacheStore.maxBytes`). Primary budget.
    public var hotBytes: Int

    /// Hot-tier entry ceiling (`PromptCacheStore.maxEntries`). Secondary safety
    /// cap — generous so bytes stay the effective budget for real caches.
    public var maxEntries: Int

    /// Cold-tier on-disk byte cap (`PromptCacheStore.coldCapBytes`).
    public var coldCapBytes: Int

    /// Master opt-in for the cold (safetensors) tier.
    public var coldEnabled: Bool

    // MARK: - Defaults

    /// 512 MiB — mirrors `Settings.default.kvCacheHotMB` (512) × 1 MiB.
    public static let defaultHotBytes = 512 * 1024 * 1024

    /// Generous secondary entry ceiling. Well above the entry count a typical
    /// `hotBytes` budget admits (MB-scale snapshots), so the byte budget stays
    /// dominant while still bounding a many-tiny-entries pathology.
    public static let defaultMaxEntries = 1024

    /// 20 GiB — mirrors `Settings.default.kvCacheColdGB` (20) × 1 GiB.
    public static let defaultColdCapBytes = 20 * 1024 * 1024 * 1024

    public init(
        hotBytes: Int = PromptCacheConfig.defaultHotBytes,
        maxEntries: Int = PromptCacheConfig.defaultMaxEntries,
        coldCapBytes: Int = PromptCacheConfig.defaultColdCapBytes,
        coldEnabled: Bool = true
    ) {
        self.hotBytes = hotBytes
        self.maxEntries = maxEntries
        self.coldCapBytes = coldCapBytes
        self.coldEnabled = coldEnabled
    }

    /// Build the runtime budget from persisted settings, converting the
    /// user-facing MB / GB knobs to bytes. Binary units (MiB / GiB) — these are
    /// memory / disk sizes. Non-positive values clamp to `0`, which drains the
    /// corresponding tier rather than misbehaving.
    public init(from settings: Settings) {
        // Clamp to sane ceilings BEFORE the ×2^20 / ×2^30 so an absurd hand-edited
        // settings.json (e.g. a billion GB) can't overflow `Int` and crash on launch.
        // The ceilings are far above any real machine (8 TiB RAM / 1 PiB disk).
        let hotMB = min(max(0, settings.kvCacheHotMB), 8_388_608)
        let coldGB = min(max(0, settings.kvCacheColdGB), 1_048_576)
        self.init(
            hotBytes: hotMB * 1024 * 1024,
            maxEntries: PromptCacheConfig.defaultMaxEntries,
            coldCapBytes: coldGB * 1024 * 1024 * 1024,
            coldEnabled: settings.kvCacheColdEnabled
        )
    }
}
