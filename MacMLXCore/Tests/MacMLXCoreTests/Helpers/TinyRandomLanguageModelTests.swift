// Copyright © 2026 macMLX. English comments only.

import MLX
import MLXLMCommon
import MLXNN
import Testing

@testable import MacMLXCore

@Suite("TinyRandomLanguageModel", .enabled(if: mlxMetallibIsAvailable, "needs the MLX Metal library"))
struct TinyRandomLanguageModelTests {

    /// A seed builds the same weights whatever random state is in use around
    /// the call. Seeding the global state let tests running in parallel draw
    /// from it between the seed and the draws, so the weights changed from
    /// run to run. An outer `withRandomState` stands in for those draws: a
    /// `make` that still drew from an ambient state would take seed 99's keys.
    @Test
    func theWeightsDoNotDependOnTheRandomStateAroundTheCall() throws {
        let reference = try weights(TinyRandomLanguageModel.qwen3.make())
        let inside = try withRandomState(MLXRandom.RandomState(seed: 99)) {
            try weights(TinyRandomLanguageModel.qwen3.make())
        }
        #expect(Set(inside.keys) == Set(reference.keys))
        for (key, value) in reference {
            guard let other = inside[key] else { continue }
            #expect(arrayEqual(value, other).item(Bool.self), "\(key)")
        }
    }

    private func weights(_ model: any LanguageModel) throws -> [String: MLXArray] {
        let module = try #require(model as? Module)
        return Dictionary(uniqueKeysWithValues: module.parameters().flattened())
    }
}
