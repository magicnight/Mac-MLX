import Testing
import Foundation
import MLXLMCommon
@testable import MacMLXCore

/// `Cohere2Model.newCache` must build the MIXED per-layer cache the interleaved
/// attention needs — a `RotatingKVCache(maxSize: sliding_window, keep: 0)` on every
/// sliding-window layer and a plain `KVCacheSimple` on every global layer (a layer
/// is global when `i % sliding_window_pattern == pattern - 1`). This mirrors the
/// Python `make_cache`, and diverges from the default `KVCacheDimensionProvider`
/// cache (uniform `KVCacheSimple`), so it is worth pinning directly.
///
/// Constructing the model builds real `Linear`/`Embedding`/`LayerNorm` parameters
/// (MLXArrays), so the suite gates on the metallib exactly like the other
/// MLX-backed suites (runs under xcodebuild; skipped under bare `swift test`).
@Suite("Cohere2 cache", .enabled(if: mlxMetallibIsAvailable, "Requires default.metallib (run under xcodebuild)"))
struct Cohere2CacheTests {

    /// A tiny 8-layer model with `sliding_window_pattern = 4`, so the expected
    /// cache-type schedule is [Rotating, Rotating, Rotating, KVCacheSimple] × 2.
    private func tinyModel(slidingWindow: Int) throws -> Cohere2Model {
        let json = """
        {
          "model_type": "cohere2",
          "vocab_size": 16,
          "hidden_size": 8,
          "num_hidden_layers": 8,
          "intermediate_size": 16,
          "num_attention_heads": 2,
          "num_key_value_heads": 1,
          "head_dim": 4,
          "sliding_window": \(slidingWindow),
          "sliding_window_pattern": 4
        }
        """
        let config = try JSONDecoder().decode(Cohere2Configuration.self, from: Data(json.utf8))
        return Cohere2Model(config)
    }

    /// The per-layer cache schedule and the rotating cache's window size must match
    /// `make_cache`: sliding layers → `RotatingKVCache(maxSize == sliding_window)`,
    /// global layers (index ≡ pattern-1 mod pattern) → `KVCacheSimple`.
    @Test("mixed cache: [Rotating, Rotating, Rotating, KV] x 2, rotating maxSize == window")
    func newCacheMixedSchedule() throws {
        let slidingWindow = 3
        let model = try tinyModel(slidingWindow: slidingWindow)
        let caches = model.newCache(parameters: nil)

        #expect(caches.count == 8)

        // Global layers are indices 3 and 7 (i % 4 == 3); all others are sliding.
        let expectedGlobal: Set<Int> = [3, 7]
        for (i, cache) in caches.enumerated() {
            if expectedGlobal.contains(i) {
                #expect(
                    cache is KVCacheSimple,
                    "layer \(i) is global and must use KVCacheSimple, got \(type(of: cache))")
            } else {
                #expect(
                    cache is RotatingKVCache,
                    "layer \(i) is sliding and must use RotatingKVCache, got \(type(of: cache))")
                #expect(
                    cache.maxSize == slidingWindow,
                    "sliding layer \(i) RotatingKVCache maxSize must equal sliding_window")
            }
        }
    }
}
