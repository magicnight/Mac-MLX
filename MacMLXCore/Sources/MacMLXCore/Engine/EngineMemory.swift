import Foundation
import MLX

// MARK: - EngineMemory

/// The one place MacMLXCore talks to MLX's allocator on behalf of an engine
/// swap, so the server stays free of `MLX` imports.
///
/// MLX does not return freed buffers to the OS; it parks them in a cache so the
/// next allocation of the same size is cheap. After a model is released that
/// cache holds the model's whole footprint (7.8 GB after dropping a 4B bf16
/// reranker, 15.5 GB after a second swap, measured in #130), and nothing else
/// in the server ever drains it. Draining it when a model is replaced trades a
/// slower first allocation for memory the process actually gives back.
public enum EngineMemory {

    /// Hand MLX's cached buffers back to the OS. Called after an engine that
    /// owned a model has been released and before its replacement loads.
    public static func releaseCachedBuffers() {
        Memory.clearCache()
    }
}
