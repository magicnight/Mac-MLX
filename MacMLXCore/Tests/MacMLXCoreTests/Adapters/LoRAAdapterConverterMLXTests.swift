import XCTest
import MLX
@testable import MacMLXCore

/// MLX-backed converter tests — `MLXArray` allocation pulls in the
/// Metal stack, which the SPM test binary does not always bundle
/// (`default.metallib` ships with Xcode-archive builds only). These
/// tests skip cleanly when run under `swift test` and execute under
/// `xcodebuild`.
final class LoRAAdapterConverterMLXTests: XCTestCase {

    /// Mirror of the helper used in `PromptCacheStoreTests` — same
    /// rationale, same skip behaviour.
    private func requireMetalOrSkip() throws {
        let bundle = Bundle(identifier: "mlx-swift_Cmlx.resources")
            ?? Bundle.allBundles.first(where: { $0.bundlePath.contains("Cmlx") })
        let metallib = bundle?.url(forResource: "default", withExtension: "metallib")
        if metallib == nil {
            throw XCTSkip("Requires default.metallib (SPM test binaries often lack it — run under xcodebuild)")
        }
    }

    func testTranslateWeightsTransposesAndRenamesLoRAPair() throws {
        try requireMetalOrSkip()

        // PEFT shape: lora_A is [rank, in], lora_B is [out, rank].
        let peftA = MLXArray([Float](repeating: 0.1, count: 8 * 4), [8, 4])  // r=8, in=4
        let peftB = MLXArray([Float](repeating: 0.2, count: 16 * 8), [16, 8]) // out=16, r=8

        let inputs: [String: MLXArray] = [
            "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": peftA,
            "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": peftB,
        ]
        let result = try LoRAAdapterConverter.translateWeights(inputs)

        XCTAssertEqual(result.arrays.count, 2)
        XCTAssertEqual(result.deepestLayer, 0)

        let aKey = "model.layers.0.self_attn.q_proj.lora_a"
        let bKey = "model.layers.0.self_attn.q_proj.lora_b"
        let outA = try XCTUnwrap(result.arrays[aKey])
        let outB = try XCTUnwrap(result.arrays[bKey])

        // Shapes are transposed.
        XCTAssertEqual(outA.shape, [4, 8], "lora_a should be [in, rank]")
        XCTAssertEqual(outB.shape, [8, 16], "lora_b should be [rank, out]")
    }

    func testTranslateWeightsDropsNonLoRAKeys() throws {
        try requireMetalOrSkip()

        let lora = MLXArray([Float](repeating: 0.1, count: 8 * 4), [8, 4])
        let stray = MLXArray([Float](repeating: 0.5, count: 16), [16])
        let inputs: [String: MLXArray] = [
            "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": lora,
            "base_model.model.modules_to_save.score.weight": stray,
        ]
        let result = try LoRAAdapterConverter.translateWeights(inputs)
        XCTAssertEqual(result.arrays.count, 1)
        XCTAssertTrue(result.arrays.keys.contains("model.layers.0.self_attn.q_proj.lora_a"))
    }

    func testTranslateWeightsTracksDeepestLayerAcrossPair() throws {
        try requireMetalOrSkip()

        let p0 = MLXArray([Float](repeating: 0.1, count: 8 * 4), [8, 4])
        let p11 = MLXArray([Float](repeating: 0.1, count: 8 * 4), [8, 4])
        let p23 = MLXArray([Float](repeating: 0.1, count: 8 * 4), [8, 4])
        let inputs: [String: MLXArray] = [
            "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": p0,
            "base_model.model.model.layers.11.self_attn.q_proj.lora_A.weight": p11,
            "base_model.model.model.layers.23.self_attn.q_proj.lora_A.weight": p23,
        ]
        let result = try LoRAAdapterConverter.translateWeights(inputs)
        XCTAssertEqual(result.deepestLayer, 23)
    }

    func testConvertPEFTAdapterWritesMLXConfigAndSafetensors() throws {
        try requireMetalOrSkip()

        let temp = try TempDir()
        let source = temp.url.appendingPathComponent("peft-source", isDirectory: true)
        let dest = temp.url.appendingPathComponent("mlx-output", isDirectory: true)
        try FileManager.default.createDirectory(at: source, withIntermediateDirectories: true)

        let peftCfg = """
        {
          "base_model_name_or_path": "Qwen3-8B-4bit",
          "r": 4,
          "lora_alpha": 8,
          "target_modules": ["q_proj", "v_proj"],
          "peft_type": "LORA"
        }
        """
        try Data(peftCfg.utf8).write(to: source.appendingPathComponent("adapter_config.json"))

        let peftA = MLXArray([Float](repeating: 0.1, count: 4 * 4), [4, 4])
        let peftB = MLXArray([Float](repeating: 0.2, count: 8 * 4), [8, 4])
        let weights: [String: MLXArray] = [
            "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": peftA,
            "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": peftB,
        ]
        try MLX.save(arrays: weights, url: source.appendingPathComponent("adapter_model.safetensors"))

        try LoRAAdapterConverter.convertPEFTAdapter(source: source, destination: dest)

        // Output config is mlx schema.
        let outCfgURL = dest.appendingPathComponent("adapter_config.json")
        let outCfg = try JSONDecoder().decode(
            LoRAAdapterConverter.MLXAdapterConfig.self,
            from: Data(contentsOf: outCfgURL)
        )
        XCTAssertEqual(outCfg.fineTuneType, "lora")
        XCTAssertEqual(outCfg.loraParameters.rank, 4)
        XCTAssertEqual(outCfg.loraParameters.scale, 2.0)        // 8 / 4
        XCTAssertEqual(outCfg.loraParameters.keys, ["q_proj", "v_proj"])
        XCTAssertEqual(outCfg.numLayers, 1)                     // inferred from layer 0

        // Output safetensors round-trips.
        let outArrays = try MLX.loadArrays(
            url: dest.appendingPathComponent("adapters.safetensors")
        )
        XCTAssertEqual(outArrays.count, 2)
        let outA = try XCTUnwrap(outArrays["model.layers.0.self_attn.q_proj.lora_a"])
        let outB = try XCTUnwrap(outArrays["model.layers.0.self_attn.q_proj.lora_b"])
        XCTAssertEqual(outA.shape, [4, 4])    // transposed from [4, 4] (square)
        XCTAssertEqual(outB.shape, [4, 8])    // transposed from [8, 4]
    }

    func testConvertThrowsForMissingConfig() throws {
        try requireMetalOrSkip()

        let temp = try TempDir()
        let source = temp.url.appendingPathComponent("empty", isDirectory: true)
        try FileManager.default.createDirectory(at: source, withIntermediateDirectories: true)
        let dest = temp.url.appendingPathComponent("dest", isDirectory: true)
        XCTAssertThrowsError(
            try LoRAAdapterConverter.convertPEFTAdapter(source: source, destination: dest)
        )
    }

    func testConvertThrowsForMissingWeights() throws {
        try requireMetalOrSkip()

        let temp = try TempDir()
        let source = temp.url.appendingPathComponent("config-only", isDirectory: true)
        try FileManager.default.createDirectory(at: source, withIntermediateDirectories: true)
        try Data(#"{"r":4}"#.utf8)
            .write(to: source.appendingPathComponent("adapter_config.json"))
        let dest = temp.url.appendingPathComponent("dest", isDirectory: true)
        XCTAssertThrowsError(
            try LoRAAdapterConverter.convertPEFTAdapter(source: source, destination: dest)
        )
    }
}

private struct TempDir {
    let url: URL
    init() throws {
        let base = FileManager.default.temporaryDirectory
            .appendingPathComponent("macmlx-lora-converter-tests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: base, withIntermediateDirectories: true)
        self.url = base
    }
}
