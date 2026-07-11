import Testing
import Foundation
@testable import MacMLXCore

/// Track F: `ModelLibraryManager.scanHuggingFaceCache(directories:)` — HF Hub
/// cache discovery. Mocks the `models--<org>--<name>/snapshots/<rev>/`
/// layout on a real temp directory (no real Hugging Face cache required).
///
/// Serialised for the same reason as the VLM/Embedder/inference suites.
@Suite("ModelLibraryManager Hugging Face cache scan", .serialized)
struct ModelLibraryManagerHuggingFaceCacheTests {

    @Test
    func discoversMLXSnapshotAsExternalReference() async throws {
        let temp = try TemporaryDirectory()
        try writeCachedModel(
            in: temp.url,
            folderName: "models--mlx-community--Qwen3-8B-4bit",
            revision: "abc123",
            files: ["config.json", "tokenizer.json", "model.safetensors"]
        )

        let mgr = ModelLibraryManager()
        let results = await mgr.scanHuggingFaceCache(directories: [temp.url])

        #expect(results.count == 1)
        #expect(results[0].id == "mlx-community/Qwen3-8B-4bit")
        #expect(results[0].displayName == "mlx-community/Qwen3-8B-4bit")
        #expect(results[0].isExternalReference == true)
        #expect(results[0].format == .mlx)
    }

    @Test
    func referencesSnapshotDirectoryInPlaceWithoutCopying() async throws {
        let temp = try TemporaryDirectory()
        let snapshotDir = try writeCachedModel(
            in: temp.url,
            folderName: "models--mlx-community--Qwen3-8B-4bit",
            revision: "abc123",
            files: ["config.json", "tokenizer.json", "model.safetensors"]
        )

        let mgr = ModelLibraryManager()
        let results = await mgr.scanHuggingFaceCache(directories: [temp.url])

        // Compare resolved paths, not raw `URL` equality: `FileManager`
        // hands back directory URLs with a trailing slash and a
        // symlink-resolved `/private/var` prefix, which differ cosmetically
        // from `snapshotDir` (built by hand via `appendingPathComponent`)
        // even though both name the exact same on-disk directory.
        #expect(
            results[0].directory.resolvingSymlinksInPath().path
            == snapshotDir.resolvingSymlinksInPath().path
        )
    }

    @Test
    func skipsNonMLXSnapshot() async throws {
        let temp = try TemporaryDirectory()
        try writeCachedModel(
            in: temp.url,
            folderName: "models--org--gguf-model",
            revision: "rev1",
            files: ["model.gguf"]
        )

        let mgr = ModelLibraryManager()
        let results = await mgr.scanHuggingFaceCache(directories: [temp.url])

        #expect(results.isEmpty)
    }

    @Test
    func skipsFoldersNotShapedLikeModelsPrefix() async throws {
        let temp = try TemporaryDirectory()
        // A dataset cache entry sitting alongside model caches — must be
        // ignored, not mistaken for a model.
        try writeCachedModel(
            in: temp.url,
            folderName: "datasets--org--some-dataset",
            revision: "rev1",
            files: ["config.json", "tokenizer.json", "model.safetensors"]
        )

        let mgr = ModelLibraryManager()
        let results = await mgr.scanHuggingFaceCache(directories: [temp.url])

        #expect(results.isEmpty)
    }

    @Test
    func toleratesMissingRootDirectory() async throws {
        let missing = FileManager.default.temporaryDirectory
            .appendingPathComponent("macmlx-hf-cache-does-not-exist-\(UUID().uuidString)")

        let mgr = ModelLibraryManager()
        let results = await mgr.scanHuggingFaceCache(directories: [missing])

        #expect(results.isEmpty)
    }

    @Test
    func oneMissingRootDoesNotBlockOtherRoots() async throws {
        let missing = FileManager.default.temporaryDirectory
            .appendingPathComponent("macmlx-hf-cache-does-not-exist-\(UUID().uuidString)")
        let temp = try TemporaryDirectory()
        try writeCachedModel(
            in: temp.url,
            folderName: "models--mlx-community--Qwen3-8B-4bit",
            revision: "abc123",
            files: ["config.json", "tokenizer.json", "model.safetensors"]
        )

        let mgr = ModelLibraryManager()
        let results = await mgr.scanHuggingFaceCache(directories: [missing, temp.url])

        #expect(results.count == 1)
        #expect(results[0].id == "mlx-community/Qwen3-8B-4bit")
    }

    @Test
    func deduplicatesMultipleSnapshotRevisionsToOnePerRepo() async throws {
        let temp = try TemporaryDirectory()
        try writeCachedModel(
            in: temp.url,
            folderName: "models--mlx-community--Qwen3-8B-4bit",
            revision: "rev-old",
            files: ["config.json", "tokenizer.json", "model.safetensors"]
        )
        try writeCachedModel(
            in: temp.url,
            folderName: "models--mlx-community--Qwen3-8B-4bit",
            revision: "rev-new",
            files: ["config.json", "tokenizer.json", "model.safetensors"]
        )

        let mgr = ModelLibraryManager()
        let results = await mgr.scanHuggingFaceCache(directories: [temp.url])

        // Same repo, multiple cached revisions on disk — exactly one
        // `LocalModel` must surface. Neither `refs/main` is present here,
        // so this also exercises the most-recently-modified fallback
        // (`writeCachedModel` writes "rev-new" after "rev-old", so it has
        // the later modification date).
        #expect(results.count == 1)
        #expect(results[0].id == "mlx-community/Qwen3-8B-4bit")
    }

    @Test
    func deduplicatesSameRepoAcrossOverlappingRoots() async throws {
        let rootA = try TemporaryDirectory()
        let rootB = try TemporaryDirectory()
        // The SAME repo cached under two configured roots (overlapping / aliased
        // cache paths) must surface exactly once — a duplicate id would break
        // SwiftUI List identity in the library view.
        for root in [rootA.url, rootB.url] {
            try writeCachedModel(
                in: root,
                folderName: "models--mlx-community--Qwen3-8B-4bit",
                revision: "abc123",
                files: ["config.json", "tokenizer.json", "model.safetensors"]
            )
        }

        let mgr = ModelLibraryManager()
        let results = await mgr.scanHuggingFaceCache(directories: [rootA.url, rootB.url])

        #expect(results.count == 1)
        #expect(results[0].id == "mlx-community/Qwen3-8B-4bit")
    }

    @Test
    func prefersRefsMainOverNewestModificationDate() async throws {
        let temp = try TemporaryDirectory()
        // "rev-newer" is written AFTER "rev-main", so it would win a
        // modification-date-only race — but refs/main pins "rev-main" as
        // the checked-out revision, and that must take priority.
        let mainSnapshot = try writeCachedModel(
            in: temp.url,
            folderName: "models--mlx-community--Qwen3-8B-4bit",
            revision: "rev-main",
            files: ["config.json", "tokenizer.json", "model.safetensors"],
            setAsMain: true
        )
        try writeCachedModel(
            in: temp.url,
            folderName: "models--mlx-community--Qwen3-8B-4bit",
            revision: "rev-newer",
            files: ["config.json", "tokenizer.json", "model.safetensors"]
        )

        let mgr = ModelLibraryManager()
        let results = await mgr.scanHuggingFaceCache(directories: [temp.url])

        #expect(results.count == 1)
        #expect(
            results[0].directory.resolvingSymlinksInPath().path
            == mainSnapshot.resolvingSymlinksInPath().path
        )
    }

    @Test
    func resolvesSizeThroughSymlinkedBlobLikeRealHFCache() async throws {
        let temp = try TemporaryDirectory()
        let modelDir = temp.url.appendingPathComponent("models--mlx-community--Qwen3-8B-4bit")
        let blobsDir = modelDir.appendingPathComponent("blobs")
        let snapshotDir = modelDir.appendingPathComponent("snapshots").appendingPathComponent("real-rev")
        try FileManager.default.createDirectory(at: blobsDir, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: snapshotDir, withIntermediateDirectories: true)

        // Real HF cache content-addresses weight blobs by sha256 and
        // symlinks them into the snapshot directory — only the
        // `.safetensors` file is blob-backed here; config.json/
        // tokenizer.json are plain files, the minimum shape
        // `ModelFormat.detect` needs.
        let blobSHA = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b85"
        let blobURL = blobsDir.appendingPathComponent(blobSHA)
        let weightBytes = Data(repeating: 0xAB, count: 123_456)
        try weightBytes.write(to: blobURL)

        let safetensorsLink = snapshotDir.appendingPathComponent("model.safetensors")
        try FileManager.default.createSymbolicLink(at: safetensorsLink, withDestinationURL: blobURL)
        try Data().write(to: snapshotDir.appendingPathComponent("config.json"))
        try Data().write(to: snapshotDir.appendingPathComponent("tokenizer.json"))

        let mgr = ModelLibraryManager()
        let results = await mgr.scanHuggingFaceCache(directories: [temp.url])

        // Proves `.fileSizeKey` resource lookups follow the symlink to the
        // blob's real size, rather than reporting the symlink's own
        // (near-zero) size or failing to resolve at all.
        #expect(results.count == 1)
        #expect(results[0].sizeBytes == Int64(weightBytes.count))
    }

    // MARK: - repoID(fromCacheFolderName:)

    @Test
    func repoIDParsesStandardCacheFolderName() {
        #expect(
            ModelLibraryManager.repoID(fromCacheFolderName: "models--mlx-community--Qwen3-8B-4bit")
            == "mlx-community/Qwen3-8B-4bit"
        )
    }

    @Test
    func repoIDReturnsNilWithoutModelsPrefix() {
        #expect(ModelLibraryManager.repoID(fromCacheFolderName: "mlx-community--Qwen3-8B-4bit") == nil)
    }

    @Test
    func repoIDReturnsNilWithoutOrgSeparator() {
        #expect(ModelLibraryManager.repoID(fromCacheFolderName: "models--justonename") == nil)
    }

    // MARK: - Helpers

    /// Writes `<root>/<folderName>/snapshots/<revision>/<files>`, mirroring
    /// HF Hub's real cache layout closely enough for the scanner (plain
    /// files here, not symlinked blobs — `buildLocalModel`'s size/config
    /// reads work identically either way since `Data(contentsOf:)` and
    /// resource-value lookups both transparently follow symlinks).
    ///
    /// - Parameter setAsMain: When `true`, also writes
    ///   `<root>/<folderName>/refs/main` containing `revision`, mirroring
    ///   how HF Hub records which snapshot is the currently checked-out
    ///   ref.
    @discardableResult
    private func writeCachedModel(
        in root: URL, folderName: String, revision: String, files: [String], setAsMain: Bool = false
    ) throws -> URL {
        let modelDir = root.appendingPathComponent(folderName)
        let snapshotDir = modelDir
            .appendingPathComponent("snapshots")
            .appendingPathComponent(revision)
        try FileManager.default.createDirectory(at: snapshotDir, withIntermediateDirectories: true)
        for file in files {
            try Data().write(to: snapshotDir.appendingPathComponent(file))
        }
        if setAsMain {
            let refsDir = modelDir.appendingPathComponent("refs")
            try FileManager.default.createDirectory(at: refsDir, withIntermediateDirectories: true)
            try Data(revision.utf8).write(to: refsDir.appendingPathComponent("main"))
        }
        return snapshotDir
    }
}

/// Auto-cleaning temp directory for filesystem-backed tests.
private struct TemporaryDirectory {
    let url: URL

    init() throws {
        let base = FileManager.default.temporaryDirectory
            .appendingPathComponent("macmlx-hf-cache-tests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: base, withIntermediateDirectories: true)
        self.url = base
    }
}
