import Testing
import Foundation
@testable import MacMLXCore

@Test
func hfModelDecodesMinimalApiPayload() throws {
    let payload = """
    {
      "id": "mlx-community/Qwen3-8B-4bit",
      "author": "mlx-community",
      "downloads": 1234,
      "likes": 56,
      "tags": ["mlx", "text-generation"]
    }
    """.data(using: .utf8)!
    let m = try JSONDecoder.huggingFace.decode(HFModel.self, from: payload)
    #expect(m.id == "mlx-community/Qwen3-8B-4bit")
    #expect(m.author == "mlx-community")
    #expect(m.downloads == 1234)
    #expect(m.likes == 56)
    #expect(m.tags.contains("mlx"))
}

@Test
func hfModelTolerantOfMissingOptionals() throws {
    let payload = """
    { "id": "x/y", "tags": [] }
    """.data(using: .utf8)!
    let m = try JSONDecoder.huggingFace.decode(HFModel.self, from: payload)
    #expect(m.id == "x/y")
    #expect(m.author == nil)
    #expect(m.downloads == nil)
}
