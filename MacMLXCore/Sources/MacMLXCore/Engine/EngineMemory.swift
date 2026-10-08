import Foundation
import MLX
import os

// MARK: - EngineMemory

/// The one place MacMLXCore talks to MLX's allocator when an engine lets go
/// of a model, a swap or an unload, so the server stays free of `MLX` imports.
///
/// MLX parks freed buffers in a cache so the next allocation of the same size
/// is cheap, and trims that cache only when it passes its limit or a single
/// allocation crosses the collection threshold; it never hands memory back on
/// its own (mlx-swift-lm's plain decode loop clears it at a generation's
/// first token and every 256 after, speculative and batched runs never do,
/// and none of that helps a model released while nothing generates). After a
/// model is released the cache holds the model's whole
/// footprint (7.8 GB after dropping a 4B bf16 reranker, 15.5 GB after a second
/// swap, measured in #130), and nothing else in the server drains it. Draining
/// it when a model is replaced trades a slower first allocation for memory the
/// process actually gives back.
enum EngineMemory {

    /// Hand MLX's cached buffers back to the OS. Called after an engine has
    /// released a model: before its replacement loads in a swap, or after an
    /// unload (#136). Never called on a path that may not have initialized MLX
    /// yet (a load that failed before reaching it): draining initializes
    /// Metal, and the SPM test job has no metallib.
    static func releaseCachedBuffers() {
        Memory.clearCache()
    }

    private static let loadedAModel = OSAllocatedUnfairLock(initialState: false)

    /// A model finished loading somewhere in this process (a chat model, an
    /// embedder, a reranker or a speech model), so Metal is up and a drain is
    /// safe even on an engine that never held a model: the pool loads every
    /// model into a fresh engine, and a load that fails after allocating there
    /// would otherwise never be drained (#136). Never set under `swift test`,
    /// where no model loads.
    static var hasLoadedAModel: Bool {
        loadedAModel.withLock { $0 }
    }

    static func noteModelLoaded() {
        loadedAModel.withLock { $0 = true }
    }
}
