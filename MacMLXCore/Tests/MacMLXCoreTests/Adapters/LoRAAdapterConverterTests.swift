import Testing
import Foundation
@testable import MacMLXCore

@Suite("LoRAAdapterConverter — pure Swift translation")
struct LoRAAdapterConverterUnitTests {

    // MARK: Key rewrite

    @Test
    func mlxKeyStripsBaseModelPrefixAndRewritesSuffixA() throws {
        let key = try LoRAAdapterConverter.mlxKey(
            forPEFTKey: "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"
        )
        #expect(key == "model.layers.0.self_attn.q_proj.lora_a")
    }

    @Test
    func mlxKeyStripsBaseModelPrefixAndRewritesSuffixB() throws {
        let key = try LoRAAdapterConverter.mlxKey(
            forPEFTKey: "base_model.model.model.layers.7.self_attn.v_proj.lora_B.weight"
        )
        #expect(key == "model.layers.7.self_attn.v_proj.lora_b")
    }

    @Test
    func mlxKeyHandlesSingleBaseModelWrapper() throws {
        let key = try LoRAAdapterConverter.mlxKey(
            forPEFTKey: "base_model.model.layers.3.mlp.gate_proj.lora_A.weight"
        )
        #expect(key == "model.layers.3.mlp.gate_proj.lora_a")
    }

    @Test
    func mlxKeyThrowsForUnrecognisedSuffix() {
        #expect(throws: LoRAAdapterConverter.Error.self) {
            _ = try LoRAAdapterConverter.mlxKey(
                forPEFTKey: "base_model.model.model.embed_tokens.weight"
            )
        }
    }

    // MARK: Layer-index extraction

    @Test
    func layerIndexFromTypicalKey() {
        #expect(LoRAAdapterConverter.layerIndex(in: "model.layers.0.self_attn.q_proj.lora_a") == 0)
        #expect(LoRAAdapterConverter.layerIndex(in: "model.layers.31.mlp.up_proj.lora_b") == 31)
    }

    @Test
    func layerIndexNilForNonLayerKey() {
        #expect(LoRAAdapterConverter.layerIndex(in: "model.embed_tokens.lora_a") == nil)
    }

    // MARK: Config translation

    @Test
    func mlxConfigurationDerivesScaleFromAlphaOverRank() {
        let peft = LocalAdapter.PEFTConfig(
            baseModelNameOrPath: "Qwen3-8B-4bit",
            r: 8,
            loraAlpha: 16,
            targetModules: ["q_proj", "v_proj"],
            peftType: "LORA"
        )
        let cfg = LoRAAdapterConverter.mlxConfiguration(from: peft, numLayers: 24)
        #expect(cfg.numLayers == 24)
        #expect(cfg.fineTuneType == "lora")
        #expect(cfg.loraParameters.rank == 8)
        #expect(cfg.loraParameters.scale == 2.0)        // 16 / 8
        #expect(cfg.loraParameters.keys == ["q_proj", "v_proj"])
    }

    @Test
    func mlxConfigurationDefaultsRankAndAlphaWhenMissing() {
        let peft = LocalAdapter.PEFTConfig(
            baseModelNameOrPath: nil,
            r: nil,
            loraAlpha: nil,
            targetModules: nil,
            peftType: nil
        )
        let cfg = LoRAAdapterConverter.mlxConfiguration(from: peft, numLayers: 1)
        #expect(cfg.loraParameters.rank == 8)            // default
        #expect(cfg.loraParameters.scale == 1.0)         // alpha default = rank → 1.0
        #expect(cfg.loraParameters.keys == nil)
    }
}
