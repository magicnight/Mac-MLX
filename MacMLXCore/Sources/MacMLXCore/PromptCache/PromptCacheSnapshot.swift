// Copyright © 2026 macMLX. English comments only.

import MLXLMCommon

/// Sendable wrapper that lets a `[any KVCache]` cross actor-isolation
/// boundaries. `KVCache` is a reference-type protocol without a `Sendable`
/// conformance in mlx-swift-lm — in practice we hand the snapshot off to the
/// generation pipeline which owns it exclusively until generation ends, so an
/// unchecked conformance is safe.
public struct PromptCacheSnapshot: @unchecked Sendable {
    public let caches: [any KVCache]
    public init(_ caches: [any KVCache]) {
        self.caches = caches
    }
}
