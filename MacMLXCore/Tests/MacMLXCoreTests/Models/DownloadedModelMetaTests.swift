import XCTest
@testable import MacMLXCore

final class DownloadedModelMetaTests: XCTestCase {
    private func tmpDir() -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appending(path: "macmlx-meta-test-\(UUID().uuidString)", directoryHint: .isDirectory)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    func testRoundtrip() throws {
        let dir = tmpDir()
        let meta = DownloadedModelMeta(
            modelID: "mlx-community/Qwen3-8B-4bit",
            commitSHA: "abc123",
            lastModifiedAtDownload: Date(timeIntervalSince1970: 1_700_000_000)
        )
        try meta.save(to: dir)
        let loaded = DownloadedModelMeta.load(from: dir)
        XCTAssertEqual(loaded?.modelID, "mlx-community/Qwen3-8B-4bit")
        XCTAssertEqual(loaded?.commitSHA, "abc123")
        XCTAssertEqual(
            loaded?.lastModifiedAtDownload?.timeIntervalSince1970 ?? 0,
            1_700_000_000,
            accuracy: 1
        )
    }

    func testMissingSidecar() {
        XCTAssertNil(DownloadedModelMeta.load(from: tmpDir()))
    }

    func testCorruptSidecar() throws {
        let dir = tmpDir()
        try "not json".data(using: .utf8)!.write(
            to: DownloadedModelMeta.url(inside: dir),
            options: .atomic
        )
        XCTAssertNil(DownloadedModelMeta.load(from: dir))
    }
}
