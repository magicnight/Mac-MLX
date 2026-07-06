import Testing
import Foundation
import MLX
import MLXNN
import MLXLLM
import MLXLMCommon
@testable import MacMLXCore

// MARK: - Minimal stub architecture

/// The smallest possible `LanguageModel` that still satisfies the protocol.
/// Exists only to prove that a type declared **outside** `mlx-swift-lm` can
/// be registered into `LLMTypeRegistry.shared` and instantiated by the
/// stock factory path. Its `callAsFunction` forward is never exercised
/// here — `createModel` only decodes config + calls the creator, which is
/// exactly the resolution step `LLMModelFactory` performs.
///
/// No `@ModuleInfo` parameters ⇒ no `MLXArray` allocation at init ⇒ the
/// test runs Metal-free under `swift test` (SPM binaries lack
/// `default.metallib`).
private final class SpikeLanguageModel: Module, LLMModel, KVCacheDimensionProvider {
    let vocabularySize: Int
    let kvHeads: [Int]

    init(_ config: SpikeConfiguration) {
        self.vocabularySize = config.vocabularySize
        self.kvHeads = Array(repeating: config.kvHeads, count: config.hiddenLayers)
        super.init()
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        // Never called in this spike — forward isn't part of the
        // registration/resolution mechanism under test.
        fatalError("SpikeLanguageModel is a registration stub; forward is not implemented")
    }

    // LoRAModel
    var loraLayers: [Module] { [] }
}

/// Matches a tiny slice of a real `config.json`.
private struct SpikeConfiguration: Codable, Sendable {
    let hiddenLayers: Int
    let vocabularySize: Int
    let kvHeads: Int

    enum CodingKeys: String, CodingKey {
        case hiddenLayers = "num_hidden_layers"
        case vocabularySize = "vocab_size"
        case kvHeads = "num_key_value_heads"
    }
}

// MARK: - Spike

/// Proves the "register a model architecture from outside mlx-swift-lm,
/// zero fork" mechanism end-to-end (minus generic weight loading, which is
/// already exercised by 54 shipped architectures).
///
/// `.serialized` because it mutates the process-global `LLMTypeRegistry
/// .shared`; a unique test-only `model_type` key avoids collision with
/// real architectures.
@Suite("ModelOverlay registration spike", .serialized)
struct ModelOverlaySpikeTests {

    /// Test-only model_type — will never appear in a real config.json.
    private static let spikeType = "macmlx_overlay_spike_test"

    private func registerSpike() async {
        await LLMTypeRegistry.shared.registerModelType(Self.spikeType) { data in
            let config = try JSONDecoder().decode(SpikeConfiguration.self, from: data)
            return SpikeLanguageModel(config)
        }
    }

    @Test
    func registryReportsCustomTypeAfterRegistration() async {
        // Precondition: stock registry doesn't know our test type.
        #expect(await LLMTypeRegistry.shared.contains(Self.spikeType) == false)

        await registerSpike()

        // The exact query the factory uses in a pre-download support check.
        #expect(await LLMTypeRegistry.shared.contains(Self.spikeType) == true)
    }

    @Test
    func factoryRegistryInstantiatesCustomTypeFromConfig() async throws {
        await registerSpike()

        let configJSON = """
        {
          "model_type": "\(Self.spikeType)",
          "num_hidden_layers": 12,
          "vocab_size": 32000,
          "num_key_value_heads": 4
        }
        """
        let data = Data(configJSON.utf8)

        // `createModel` is precisely what `LLMModelFactory._load` calls to
        // turn (config data, model_type) into a `LanguageModel`. Driving it
        // directly proves the factory's resolution path reaches our creator.
        let model = try await LLMTypeRegistry.shared.createModel(
            configuration: data,
            modelType: Self.spikeType
        )

        let spike = try #require(model as? SpikeLanguageModel)
        #expect(spike.vocabularySize == 32000)
        #expect(spike.kvHeads == Array(repeating: 4, count: 12))
    }

    @Test
    func unknownTypeStillThrows() async {
        // Sanity: registration is additive, not a blanket "accept anything".
        await #expect(throws: (any Error).self) {
            _ = try await LLMTypeRegistry.shared.createModel(
                configuration: Data("{}".utf8),
                modelType: "macmlx_definitely_not_registered_xyz"
            )
        }
    }

    @Test
    func modelOverlayRegisterAllIsCallableAndIdempotent() async {
        // The production hook compiles + runs. No-op today, but the
        // plumbing MLXSwiftEngine relies on is exercised.
        await ModelOverlay.registerAll()
        await ModelOverlay.registerAll()
    }
}
