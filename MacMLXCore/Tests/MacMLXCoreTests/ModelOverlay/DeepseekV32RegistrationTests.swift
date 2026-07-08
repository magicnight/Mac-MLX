import Testing
import Foundation
import MLX
import MLXNN
import MLXLLM
import MLXLMCommon
@testable import MacMLXCore

/// Proves `ModelOverlay.registerAll()` wires the pure-Swift DeepSeek V3.2
/// architecture into the stock `LLMTypeRegistry.shared`, so the factory
/// resolves `config.json`'s `model_type: deepseek_v32` to `DeepseekV32Model`
/// with no fork. The generic register-from-outside mechanism (and the
/// unknown-type-still-throws negative case) is already proven by
/// `ModelOverlaySpikeTests`; this pins the concrete production registration.
///
/// `.serialized` because it touches the process-global registry. The registry
/// is a shared singleton that other suites also register into, so the
/// assertions are post-conditions (`contains == true` after `registerAll`),
/// which hold regardless of cross-suite ordering — asserting a `false`
/// pre-condition here would be racy.
@Suite("DeepseekV32 overlay registration", .serialized)
struct DeepseekV32RegistrationTests {

    private static let modelType = "deepseek_v32"

    @Test
    func registerAllRegistersDeepseekV32() async {
        await ModelOverlay.registerAll()
        // The exact query the factory uses in a pre-download support check.
        #expect(await LLMTypeRegistry.shared.contains(Self.modelType) == true)
    }

    @Test
    func registerAllIsIdempotent() async {
        // Double-call must not throw or leave the registry in a bad state —
        // MLXSwiftEngine may invoke `registerAll` more than once.
        await ModelOverlay.registerAll()
        await ModelOverlay.registerAll()
        #expect(await LLMTypeRegistry.shared.contains(Self.modelType) == true)
    }

    @Test(.enabled(if: mlxMetallibIsAvailable, "Requires default.metallib (run under xcodebuild)"))
    func factoryInstantiatesDeepseekV32FromConfig() async throws {
        // Instantiating DeepseekV32Model creates real MLXArrays (Linear /
        // Embedding inits) — fatal under bare `swift test` without the
        // metallib, hence the trait gate. The two registry-only tests above
        // never invoke the creator, so they stay ungated.
        await ModelOverlay.registerAll()

        // A minimal `config.json` slice; `createModel` is exactly what
        // `LLMModelFactory` calls to turn (config data, model_type) into a
        // `LanguageModel`, so driving it proves the resolution path reaches
        // our creator and decodes via `JSONDecoder.json5()`.
        let configJSON = """
        {
          "model_type": "deepseek_v32",
          "vocab_size": 32,
          "hidden_size": 16,
          "num_hidden_layers": 1,
          "num_attention_heads": 2,
          "num_key_value_heads": 2
        }
        """
        let model = try await LLMTypeRegistry.shared.createModel(
            configuration: Data(configJSON.utf8),
            modelType: Self.modelType
        )
        #expect(model is DeepseekV32Model)
    }
}
