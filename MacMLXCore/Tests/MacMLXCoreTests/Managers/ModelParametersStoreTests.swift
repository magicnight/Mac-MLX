import Testing
import Foundation
@testable import MacMLXCore

// Wrapped in a @Suite struct so module-scope function names don't clash
// with other stores' tests.
@Suite
struct ModelParametersStoreTests {

private func makeTempStore() -> (ModelParametersStore, URL) {
    let dir = URL(filePath: NSTemporaryDirectory(), directoryHint: .isDirectory)
        .appending(path: "macmlx-params-\(UUID().uuidString)", directoryHint: .isDirectory)
    return (ModelParametersStore(directory: dir), dir)
}

// MARK: - Round-trip

@Test
func savedParametersRoundTrip() async throws {
    let (store, dir) = makeTempStore()
    defer { try? FileManager.default.removeItem(at: dir) }

    let params = ModelParameters(
        temperature: 0.42,
        topP: 0.88,
        maxTokens: 4096,
        systemPrompt: "You are a grumpy code reviewer."
    )
    try await store.save(params, for: "mlx-community/Qwen3-8B-4bit")
    let loaded = await store.load(for: "mlx-community/Qwen3-8B-4bit")
    #expect(loaded.temperature == 0.42)
    #expect(loaded.topP == 0.88)
    #expect(loaded.maxTokens == 4096)
    #expect(loaded.systemPrompt == "You are a grumpy code reviewer.")
}

@Test
func loadReturnsDefaultWhenNoFileExists() async throws {
    let (store, dir) = makeTempStore()
    defer { try? FileManager.default.removeItem(at: dir) }

    let loaded = await store.load(for: "never-saved-this-model")
    #expect(loaded == .default)
}

@Test
func resetRemovesStoredOverrides() async throws {
    let (store, dir) = makeTempStore()
    defer { try? FileManager.default.removeItem(at: dir) }

    let custom = ModelParameters(temperature: 1.5, topP: 0.5, maxTokens: 100, systemPrompt: "x")
    try await store.save(custom, for: "m1")
    await store.reset(for: "m1")
    let loaded = await store.load(for: "m1")
    #expect(loaded == .default)
}

@Test
func modelIDsWithSlashesArePathSafe() async throws {
    let (store, dir) = makeTempStore()
    defer { try? FileManager.default.removeItem(at: dir) }

    // Hugging Face-style IDs have a literal "/" — it must NOT create a
    // subdirectory, or `load` won't be able to find the file.
    let slash = "mlx-community/Llama-3.2-3B-4bit"
    let dash = "mlx-community-Llama-3.2-3B-4bit"
    let params = ModelParameters(temperature: 0.1, topP: 1.0, maxTokens: 64, systemPrompt: "s")
    try await store.save(params, for: slash)
    let loaded = await store.load(for: slash)
    #expect(loaded.temperature == 0.1)
    // Different IDs with same slug must NOT collide.
    let loaded2 = await store.load(for: dash)
    #expect(loaded2 == .default)
}

@Test
func corruptFileFallsBackToDefault() async throws {
    let (store, dir) = makeTempStore()
    defer { try? FileManager.default.removeItem(at: dir) }

    try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    let url = dir.appending(path: "bad.json", directoryHint: .notDirectory)
    try Data("{this is not valid json".utf8).write(to: url)

    let loaded = await store.load(for: "bad")
    #expect(loaded == .default)
}

// MARK: - v0.5.1 fields (alias / ttlSeconds / templateKwargs)

@Test
func newOptionalFieldsRoundTrip() async throws {
    let (store, dir) = makeTempStore()
    defer { try? FileManager.default.removeItem(at: dir) }

    let params = ModelParameters(
        temperature: 0.5,
        topP: 0.9,
        maxTokens: 512,
        systemPrompt: "s",
        alias: "qwen",
        ttlSeconds: 900,
        templateKwargs: ["enable_thinking": .bool(true)]
    )
    try await store.save(params, for: "mlx-community/Qwen3-8B-4bit")
    let loaded = await store.load(for: "mlx-community/Qwen3-8B-4bit")
    #expect(loaded.alias == "qwen")
    #expect(loaded.ttlSeconds == 900)
    #expect(loaded.templateKwargs == ["enable_thinking": .bool(true)])
}

@Test
func newOptionalFieldsDefaultToNil() async throws {
    // A pre-v0.5.1 file (only the four original keys) must still load,
    // with the new fields defaulting to nil.
    let (store, dir) = makeTempStore()
    defer { try? FileManager.default.removeItem(at: dir) }

    try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    let url = dir.appending(path: "legacy.json", directoryHint: .notDirectory)
    let legacy = """
    {"temperature":0.7,"topP":0.95,"maxTokens":2048,"systemPrompt":"hi"}
    """
    try Data(legacy.utf8).write(to: url)

    let loaded = await store.load(for: "legacy")
    #expect(loaded.alias == nil)
    #expect(loaded.ttlSeconds == nil)
    #expect(loaded.templateKwargs == nil)
}

@Test
func modelIDForAliasFindsMatch() async throws {
    let (store, dir) = makeTempStore()
    defer { try? FileManager.default.removeItem(at: dir) }

    let slashID = "mlx-community/Qwen3-8B-4bit"
    var params = ModelParameters.default
    params.alias = "qwen"
    try await store.save(params, for: slashID)

    let resolved = await store.modelID(forAlias: "qwen")
    // The decoded filename must recover the original slashed id.
    #expect(resolved == slashID)
}

@Test
func modelIDForAliasReturnsNilForUnknownAlias() async throws {
    let (store, dir) = makeTempStore()
    defer { try? FileManager.default.removeItem(at: dir) }

    var params = ModelParameters.default
    params.alias = "qwen"
    try await store.save(params, for: "some-model")

    let resolved = await store.modelID(forAlias: "not-a-known-alias")
    #expect(resolved == nil)
}

@Test
func modelIDForAliasIgnoresEmptyQuery() async throws {
    let (store, dir) = makeTempStore()
    defer { try? FileManager.default.removeItem(at: dir) }

    // A stored empty alias must not be resolvable by an empty query.
    var params = ModelParameters.default
    params.alias = ""
    try await store.save(params, for: "some-model")

    let resolved = await store.modelID(forAlias: "")
    #expect(resolved == nil)
}

} // end @Suite ModelParametersStoreTests
