// ModelLibrarySearchStateTests.swift
// macMLXTests
//
// The Hugging Face search state in `ModelLibraryViewModel`: whatever the
// latest `searchHF()` asked for is what the spinner shows, however the
// earlier searches it cancelled were timed. A stalling URLProtocol holds a
// search in flight for as long as a test needs it to.

import Foundation
import Pulse
import Testing
import MacMLXCore
@testable import macMLX

/// Answers `/api/models?search=…` with one model, except that a search whose
/// query starts with `stall` never answers until its task is cancelled, and
/// `/api/models/<id>` with one file of 1,234 bytes, which is what the size
/// enrichment reads. Anything else gets a 404.
nonisolated final class StallingHubProtocol: URLProtocol {
    private static let lock = NSLock()
    nonisolated(unsafe) private static var _stalledQueries: [String] = []

    static var stalledQueries: [String] { lock.withLock { _stalledQueries } }
    static func reset() { lock.withLock { _stalledQueries = [] } }

    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

    override func startLoading() {
        guard let url = request.url, let client else { return }
        let items = URLComponents(url: url, resolvingAgainstBaseURL: false)?.queryItems ?? []
        if url.path == "/api/models", let query = items.first(where: { $0.name == "search" })?.value {
            if query.hasPrefix("stall") {
                Self.lock.withLock { Self._stalledQueries.append(query) }
                return
            }
            let body = Data(#"[{"id":"mlx-community/\#(query)-4bit","tags":[]}]"#.utf8)
            let response = HTTPURLResponse(url: url, statusCode: 200, httpVersion: nil, headerFields: nil)
            if let response { client.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed) }
            client.urlProtocol(self, didLoad: body)
            client.urlProtocolDidFinishLoading(self)
            return
        }
        if url.path.hasPrefix("/api/models/") {
            let body = Data(#"{"siblings":[{"rfilename":"model.safetensors","size":1234}]}"#.utf8)
            let response = HTTPURLResponse(url: url, statusCode: 200, httpVersion: nil, headerFields: nil)
            if let response { client.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed) }
            client.urlProtocol(self, didLoad: body)
            client.urlProtocolDidFinishLoading(self)
            return
        }
        if let response = HTTPURLResponse(url: url, statusCode: 404, httpVersion: nil, headerFields: nil) {
            client.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        }
        client.urlProtocolDidFinishLoading(self)
    }

    override func stopLoading() {}
}

@Suite("ModelLibraryViewModel search state", .serialized)
@MainActor
final class ModelLibrarySearchStateTests {

    private let viewModel: ModelLibraryViewModel
    /// Holds the size cache's file and the log store, so nothing lands in
    /// `~/.mac-mlx`. Removed in `deinit`.
    private let scratch: URL

    init() throws {
        StallingHubProtocol.reset()
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [StallingHubProtocol.self]
        scratch = FileManager.default.temporaryDirectory
            .appendingPathComponent("macmlx-search-state-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: scratch, withIntermediateDirectories: true)
        let sizes = scratch.appendingPathComponent("sizes.json")
        let logs = try LoggerStore(
            storeURL: scratch.appendingPathComponent("logs.pulse"), options: [.create])
        viewModel = ModelLibraryViewModel(
            library: ModelLibraryManager(),
            coordinator: EngineCoordinator(logs: LogManager(store: logs)),
            downloader: HFDownloader(urlSession: URLSession(configuration: configuration)),
            sizeCache: HFSizeCache(fileURL: sizes),
            modelDirectoryProvider: { FileManager.default.temporaryDirectory },
            hfCacheSettingsProvider: { (false, []) }
        )
    }

    deinit {
        try? FileManager.default.removeItem(at: scratch)
    }

    /// Polls until `condition` holds, for up to 60 s.
    private func waitFor(_ condition: () -> Bool) async throws {
        let deadline = ContinuousClock.now + .seconds(60)
        while !condition() {
            #expect(ContinuousClock.now < deadline)
            if ContinuousClock.now >= deadline { return }
            try await Task.sleep(for: .milliseconds(10))
        }
    }

    @Test
    func clearingTheFieldWhileASearchIsInFlightStopsTheSpinner() async throws {
        viewModel.searchQuery = "stall-cleared"
        viewModel.searchHF()
        try await waitFor { StallingHubProtocol.stalledQueries.contains("stall-cleared") }
        #expect(viewModel.isSearchingHF)

        viewModel.searchQuery = ""
        viewModel.searchHF()
        #expect(!viewModel.isSearchingHF)

        // The cancelled search must not put it back.
        try await Task.sleep(for: .milliseconds(300))
        #expect(!viewModel.isSearchingHF)
        #expect(viewModel.hfModels.isEmpty)
        #expect(viewModel.hfError == nil)
    }

    @Test
    func clearingTheFieldBeforeTheSearchTaskRunsStopsTheSpinnerToo() async throws {
        // No suspension between the two calls, so the first search's task is
        // cancelled before it has run at all.
        viewModel.searchQuery = "stall-never-ran"
        viewModel.searchHF()
        viewModel.searchQuery = ""
        viewModel.searchHF()
        #expect(!viewModel.isSearchingHF)

        try await Task.sleep(for: .milliseconds(300))
        #expect(!viewModel.isSearchingHF)
    }

    @Test
    func aSearchThatSupersedesAStalledOneEndsTheSpinnerWithItsOwnResults() async throws {
        viewModel.searchQuery = "stall-superseded"
        viewModel.searchHF()
        try await waitFor { StallingHubProtocol.stalledQueries.contains("stall-superseded") }

        viewModel.searchQuery = "llama"
        viewModel.searchHF()
        #expect(viewModel.isSearchingHF)
        try await waitFor { !viewModel.isSearchingHF }

        #expect(viewModel.hfModels.map(\.id) == ["mlx-community/llama-4bit"])
        #expect(viewModel.hfError == nil)
    }

    /// One result to size, fewer than the enrichment's four concurrent
    /// fetches. The fill loop used to wait for four fetches to start and
    /// spun the main actor forever once the list ran out (#143). If it
    /// regresses, this test hangs rather than fails: nothing else runs on
    /// the main actor to time it out.
    @Test
    func aSearchWithFewerResultsToSizeThanTheFetchCapGetsItsSizes() async throws {
        viewModel.searchQuery = "llama"
        viewModel.searchHF()
        try await waitFor { viewModel.hfModels.first?.sizeBytes != nil }

        #expect(viewModel.hfModels.map(\.sizeBytes) == [1234])
        #expect(!viewModel.isSearchingHF)
    }
}
