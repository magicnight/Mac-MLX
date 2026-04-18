import XCTest

@testable import MacMLXCore

final class MLXSwiftEnginePreflightTests: XCTestCase {

    private func writeConfig(_ json: String) throws -> URL {
        let tmp = FileManager.default.temporaryDirectory
            .appending(
                path: "macmlx-preflight-\(UUID().uuidString)", directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: tmp, withIntermediateDirectories: true)
        let url = tmp.appending(path: "config.json", directoryHint: .notDirectory)
        try json.write(to: url, atomically: true, encoding: .utf8)
        return url
    }

    func testGemma4MoEFlaggedUnsupported() throws {
        let config = """
            {"model_type":"gemma4_text","num_experts":8,"num_hidden_layers":26}
            """
        let url = try writeConfig(config)
        XCTAssertTrue(MLXSwiftEngine.isUnsupportedGemma4MoE(configURL: url))
    }

    func testGemma4DenseNotFlagged() throws {
        let config = """
            {"model_type":"gemma4_text","num_hidden_layers":26}
            """
        let url = try writeConfig(config)
        XCTAssertFalse(MLXSwiftEngine.isUnsupportedGemma4MoE(configURL: url))
    }

    func testNonGemma4MoENotFlagged() throws {
        // Qwen, Mixtral, etc. have MoE but upstream supports them — we
        // only want to reject Gemma 4 MoE specifically.
        let config = """
            {"model_type":"mixtral","num_local_experts":8}
            """
        let url = try writeConfig(config)
        XCTAssertFalse(MLXSwiftEngine.isUnsupportedGemma4MoE(configURL: url))
    }

    func testNestedTextConfig() throws {
        // Gemma 4 vision-language configs nest the text fields under
        // "text_config".
        let config = """
            {"model_type":"gemma4","text_config":{"model_type":"gemma4_text","num_experts":8}}
            """
        let url = try writeConfig(config)
        XCTAssertTrue(MLXSwiftEngine.isUnsupportedGemma4MoE(configURL: url))
    }

    func testMissingConfigNotFlagged() {
        let bogus = URL(filePath: "/tmp/macmlx-nonexistent-\(UUID().uuidString)")
        XCTAssertFalse(MLXSwiftEngine.isUnsupportedGemma4MoE(configURL: bogus))
    }
}
