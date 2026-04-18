import Testing
import Foundation
import os
@testable import MacMLXCore

// MARK: - MockURLProtocol

/// URLProtocol subclass that intercepts all requests and routes them through a handler closure.
/// Inject via `URLSessionConfiguration.protocolClasses` to avoid real network calls.
///
/// The handler is stored under an OSAllocatedUnfairLock (instead of
/// `nonisolated(unsafe) static var`) because URLSession invokes
/// `startLoading()` on its own background executor, and the Swift 6.0+
/// runtime asserts on cross-executor access to unsynchronised statics.
final class MockURLProtocol: URLProtocol {
    typealias Handler = @Sendable (URLRequest) -> (HTTPURLResponse, Data)

    private static let lockedHandler = OSAllocatedUnfairLock<Handler?>(initialState: nil)

    static var handler: Handler? {
        get { lockedHandler.withLock { $0 } }
        set { lockedHandler.withLock { $0 = newValue } }
    }

    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

    override func startLoading() {
        guard let handler = Self.handler else {
            client?.urlProtocol(self, didFailWithError: URLError(.badServerResponse))
            return
        }
        let (response, data) = handler(request)
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        client?.urlProtocol(self, didLoad: data)
        client?.urlProtocolDidFinishLoading(self)
    }

    override func stopLoading() {}
}

// MARK: - Helpers

private func makeMockSession() -> URLSession {
    let config = URLSessionConfiguration.ephemeral
    config.protocolClasses = [MockURLProtocol.self]
    return URLSession(configuration: config)
}

private func httpResponse(url: URL, statusCode: Int) -> HTTPURLResponse {
    HTTPURLResponse(url: url, statusCode: statusCode, httpVersion: nil, headerFields: nil)!
}

// MARK: - Test Suite (serialized to avoid static-handler races)

/// HFDownloader tests are serialized because MockURLProtocol.handler is a shared static.
@Suite(.serialized)
struct HFDownloaderTests {

    @Test func searchDecodesMockResponse() async throws {
        let json = """
        [
          {"id": "mlx-community/Qwen3-8B-4bit", "downloads": 1000, "likes": 42, "tags": ["mlx"]},
          {"id": "mlx-community/Llama-3-8B-Instruct-4bit", "downloads": 500, "likes": 10, "tags": []}
        ]
        """.data(using: .utf8)!

        MockURLProtocol.handler = { request in
            let url = request.url ?? URL(string: "https://example.com")!
            return (httpResponse(url: url, statusCode: 200), json)
        }
        defer { MockURLProtocol.handler = nil }

        let downloader = HFDownloader(urlSession: makeMockSession())
        let results = try await downloader.search(query: "qwen")

        #expect(results.count == 2)
        #expect(results[0].id == "mlx-community/Qwen3-8B-4bit")
        #expect(results[1].id == "mlx-community/Llama-3-8B-Instruct-4bit")
    }

    @Test func filesDecodesSiblingsArray() async throws {
        let json = """
        {
          "id": "mlx-community/Qwen3-8B-4bit",
          "siblings": [
            {"rfilename": "config.json"},
            {"rfilename": "model.safetensors", "size": 4294967296}
          ]
        }
        """.data(using: .utf8)!

        MockURLProtocol.handler = { request in
            let url = request.url ?? URL(string: "https://example.com")!
            return (httpResponse(url: url, statusCode: 200), json)
        }
        defer { MockURLProtocol.handler = nil }

        let downloader = HFDownloader(urlSession: makeMockSession())
        let files = try await downloader.files(for: "mlx-community/Qwen3-8B-4bit")

        #expect(files.count == 2)
        #expect(files[0].path == "config.json")
        #expect(files[1].path == "model.safetensors")
        #expect(files[1].size == 4_294_967_296)
    }

    @Test func searchPropagates404AsError() async throws {
        MockURLProtocol.handler = { request in
            let url = request.url ?? URL(string: "https://example.com")!
            return (httpResponse(url: url, statusCode: 404), Data())
        }
        defer { MockURLProtocol.handler = nil }

        let downloader = HFDownloader(urlSession: makeMockSession())

        do {
            _ = try await downloader.search(query: "nonexistent-model-xyz")
            Issue.record("Expected an error to be thrown")
        } catch let error as DownloadError {
            if case .badStatusCode(let code, _) = error {
                #expect(code == 404)
            } else {
                Issue.record("Expected badStatusCode(404), got \(error)")
            }
        }
    }

    @Test func searchPropagates500AsError() async throws {
        MockURLProtocol.handler = { request in
            let url = request.url ?? URL(string: "https://example.com")!
            return (httpResponse(url: url, statusCode: 500), Data())
        }
        defer { MockURLProtocol.handler = nil }

        let downloader = HFDownloader(urlSession: makeMockSession())

        do {
            _ = try await downloader.search(query: "test")
            Issue.record("Expected an error to be thrown")
        } catch let error as DownloadError {
            if case .badStatusCode(let code, _) = error {
                #expect(code == 500)
            } else {
                Issue.record("Expected badStatusCode(500), got \(error)")
            }
        }
    }

    @Test func sizeBytesUsesDeclaredSiblings() async throws {
        // Siblings all have explicit sizes — sizeBytes should sum them
        // directly and never HEAD anything.
        let json = """
        {
          "id": "mlx-community/all-declared",
          "siblings": [
            {"rfilename": "config.json", "size": 512},
            {"rfilename": "tokenizer.json", "size": 2048},
            {"rfilename": "model.safetensors", "size": 4294967296}
          ]
        }
        """.data(using: .utf8)!

        // Count requests so we can assert no HEAD traffic was issued.
        let headCount = OSAllocatedUnfairLock<Int>(initialState: 0)
        MockURLProtocol.handler = { request in
            let url = request.url ?? URL(string: "https://example.com")!
            if request.httpMethod == "HEAD" {
                headCount.withLock { $0 += 1 }
            }
            return (httpResponse(url: url, statusCode: 200), json)
        }
        defer { MockURLProtocol.handler = nil }

        let downloader = HFDownloader(urlSession: makeMockSession())
        let total = try await downloader.sizeBytes(for: "mlx-community/all-declared")

        #expect(total == 512 + 2048 + 4_294_967_296)
        #expect(headCount.withLock { $0 } == 0)
    }

    @Test func filesSingleSiblingNoSize() async throws {
        let json = """
        {
          "id": "mlx-community/tiny-model",
          "siblings": [
            {"rfilename": "config.json"}
          ]
        }
        """.data(using: .utf8)!

        MockURLProtocol.handler = { request in
            let url = request.url ?? URL(string: "https://example.com")!
            return (httpResponse(url: url, statusCode: 200), json)
        }
        defer { MockURLProtocol.handler = nil }

        let downloader = HFDownloader(urlSession: makeMockSession())
        let files = try await downloader.files(for: "mlx-community/tiny-model")

        #expect(files.count == 1)
        #expect(files[0].path == "config.json")
        #expect(files[0].size == nil)
    }
}
