import Testing
import Foundation
@testable import MacMLXCore

@Suite("LocalAdapter")
struct LocalAdapterTests {

    @Test
    func decodesPEFTAdapterConfig() throws {
        let json = """
        {
          "base_model_name_or_path": "mlx-community/Qwen3-8B-4bit",
          "r": 8,
          "lora_alpha": 16,
          "target_modules": ["q_proj", "v_proj"],
          "peft_type": "LORA"
        }
        """
        let cfg = try JSONDecoder().decode(LocalAdapter.PEFTConfig.self, from: Data(json.utf8))
        #expect(cfg.baseModelNameOrPath == "mlx-community/Qwen3-8B-4bit")
        #expect(cfg.r == 8)
        #expect(cfg.loraAlpha == 16)
        #expect(cfg.targetModules == ["q_proj", "v_proj"])
        #expect(cfg.peftType == "LORA")
    }

    @Test
    func decodesAdapterConfigMissingOptionalFields() throws {
        let json = #"{"r": 4}"#
        let cfg = try JSONDecoder().decode(LocalAdapter.PEFTConfig.self, from: Data(json.utf8))
        #expect(cfg.r == 4)
        #expect(cfg.baseModelNameOrPath == nil)
        #expect(cfg.loraAlpha == nil)
        #expect(cfg.targetModules == nil)
        #expect(cfg.peftType == nil)
    }

    @Test
    func roundTripsThroughJSON() throws {
        let original = LocalAdapter(
            name: "qwen3-medical-lora",
            directory: URL(fileURLWithPath: "/tmp/medical"),
            targetModel: "mlx-community/Qwen3-8B-4bit",
            rank: 8,
            targetModules: ["q_proj", "v_proj"]
        )
        let data = try JSONEncoder().encode(original)
        let back = try JSONDecoder().decode(LocalAdapter.self, from: data)
        #expect(back == original)
    }

    @Test
    func idMirrorsName() {
        let a = LocalAdapter(
            name: "x",
            directory: URL(fileURLWithPath: "/tmp/x"),
            targetModel: nil,
            rank: nil,
            targetModules: []
        )
        #expect(a.id == "x")
    }
}
