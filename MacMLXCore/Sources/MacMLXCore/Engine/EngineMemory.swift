import Foundation
import MLX

// MARK: - EngineMemory

/// The one place MacMLXCore talks to MLX's allocator on behalf of an engine
/// swap, so the server stays free of `MLX` imports.
///
/// MLX parks freed buffers in a cache so the next allocation of the same size
/// is cheap, and trims that cache only when it passes its limit or a single
/// allocation crosses the collection threshold; it never hands memory back on
/// its own. After a model is released the cache holds the model's whole
/// footprint (7.8 GB after dropping a 4B bf16 reranker, 15.5 GB after a second
/// swap, measured in #130), and nothing else in the server drains it. Draining
/// it when a model is replaced trades a slower first allocation for memory the
/// process actually gives back.
enum EngineMemory {

    /// Hand MLX's cached buffers back to the OS. Called after an engine that
    /// owned a model has been released and before its replacement loads. Never
    /// called on a path that may not have initialized MLX yet (a failed load):
    /// draining initializes Metal, and the SPM test job has no metallib.
    static func releaseCachedBuffers() {
        Memory.clearCache()
    }
}
