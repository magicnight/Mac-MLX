import Testing
import Foundation
@testable import MacMLXCore

// MARK: - Test doubles

/// Scriptable stand-in for the engine's `generate` closure. Each call vends the
/// next turn's chunks and records the request, so tests assert message-list
/// evolution and how many generations ran. `@unchecked Sendable`: the mutable
/// state is guarded by an `NSLock`.
private final class ScriptedGenerate: @unchecked Sendable {
    private let turns: [[GenerateChunk]]
    private let lock = NSLock()
    private var index = 0
    private var recorded: [GenerateRequest] = []

    init(_ turns: [[GenerateChunk]]) { self.turns = turns }

    var callCount: Int {
        lock.lock(); defer { lock.unlock() }
        return index
    }

    var requests: [GenerateRequest] {
        lock.lock(); defer { lock.unlock() }
        return recorded
    }

    func generate(_ request: GenerateRequest) -> AsyncThrowingStream<GenerateChunk, Error> {
        lock.lock()
        let i = index
        index += 1
        recorded.append(request)
        let chunks = i < turns.count ? turns[i] : []
        lock.unlock()
        return AsyncThrowingStream { continuation in
            for chunk in chunks { continuation.yield(chunk) }
            continuation.finish()
        }
    }
}

/// One-shot gate so a test can wait until a stubbed `callTool` is actually
/// running before it cancels.
private actor CallGate {
    private var opened = false
    private var waiters: [CheckedContinuation<Void, Never>] = []

    func open() {
        opened = true
        for waiter in waiters { waiter.resume() }
        waiters.removeAll()
    }

    func wait() async {
        if opened { return }
        await withCheckedContinuation { waiters.append($0) }
    }
}

private func toolCallChunk(_ call: ToolCallRequest) -> GenerateChunk {
    GenerateChunk(text: "", finishReason: .toolCalls, toolCalls: [call])
}

// MARK: - Tests

@Suite("ToolCallingSession")
struct ToolCallingSessionTests {

    @Test("happy path: one tool call, then a final answer")
    func happyPath() async throws {
        let call = ToolCallRequest(
            id: "call_1", name: "get_weather", arguments: ["city": .string("Paris")])
        let scripted = ScriptedGenerate([
            [toolCallChunk(call)],
            [GenerateChunk(text: "It's sunny.", finishReason: .stop)],
        ])
        let session = ToolCallingSession(
            generate: { scripted.generate($0) },
            callTool: { server, name, args in
                #expect(server == "weather-srv")
                #expect(name == "get_weather")
                #expect(args?["city"] == JSONValue.string("Paris"))
                return "22C, sunny"
            },
            toolIndex: ["get_weather": "weather-srv"]
        )

        var events: [ToolLoopEvent] = []
        for try await event in session.run(request("weather?")) { events.append(event) }

        #expect(events.count == 5)
        guard case .assistantDelta = events[0] else { Issue.record("event 0"); return }
        guard case .toolCallStarted(let started, let server) = events[1] else {
            Issue.record("event 1"); return
        }
        #expect(started.name == "get_weather")
        #expect(server == "weather-srv")
        guard case .toolResult(let id, let content, let isError) = events[2] else {
            Issue.record("event 2"); return
        }
        #expect(id == "call_1")
        #expect(content == "22C, sunny")
        #expect(isError == false)
        guard case .assistantDelta(let finalChunk) = events[3] else { Issue.record("event 3"); return }
        #expect(finalChunk.text == "It's sunny.")
        guard case .finished(let reason) = events[4] else { Issue.record("event 4"); return }
        #expect(reason == .stop)

        // Message-list evolution: the second turn sees user + assistant(calls) + tool.
        #expect(scripted.callCount == 2)
        let secondTurn = scripted.requests[1].messages
        #expect(secondTurn.count == 3)
        #expect(secondTurn[0].role == .user)
        #expect(secondTurn[1].role == .assistant)
        #expect(secondTurn[1].toolCalls?.first?.name == "get_weather")
        #expect(secondTurn[2].role == .tool)
        #expect(secondTurn[2].toolCallID == "call_1")
        #expect(secondTurn[2].content == "22C, sunny")
    }

    @Test("unknown tool yields an error result fed back; loop continues")
    func unknownTool() async throws {
        let call = ToolCallRequest(id: "call_1", name: "mystery", arguments: [:])
        let scripted = ScriptedGenerate([
            [toolCallChunk(call)],
            [GenerateChunk(text: "done", finishReason: .stop)],
        ])
        let session = ToolCallingSession(
            generate: { scripted.generate($0) },
            callTool: { _, _, _ in
                Issue.record("callTool must not run for an unknown tool")
                return ""
            },
            toolIndex: [:]  // no server provides "mystery"
        )

        var events: [ToolLoopEvent] = []
        for try await event in session.run(request("go")) { events.append(event) }

        let results = events.compactMap { event -> (String, Bool)? in
            if case .toolResult(_, let content, let isError) = event { return (content, isError) }
            return nil
        }
        #expect(results.count == 1)
        #expect(results[0].1 == true)
        #expect(results[0].0.contains("mystery"))

        let started = events.contains { if case .toolCallStarted = $0 { return true }; return false }
        #expect(started == false)

        #expect(scripted.callCount == 2)  // loop continued to a final answer
        #expect(scripted.requests[1].messages.last?.role == .tool)
        #expect(scripted.requests[1].messages.last?.toolCallID == "call_1")
    }

    @Test("tool throw becomes an error result; loop continues")
    func toolThrows() async throws {
        struct Boom: Error {}
        let call = ToolCallRequest(id: "call_1", name: "flaky", arguments: [:])
        let scripted = ScriptedGenerate([
            [toolCallChunk(call)],
            [GenerateChunk(text: "recovered", finishReason: .stop)],
        ])
        let session = ToolCallingSession(
            generate: { scripted.generate($0) },
            callTool: { _, _, _ in throw Boom() },
            toolIndex: ["flaky": "srv"]
        )

        var events: [ToolLoopEvent] = []
        for try await event in session.run(request("go")) { events.append(event) }

        let errored = events.compactMap { event -> Bool? in
            if case .toolResult(_, _, let isError) = event { return isError }
            return nil
        }
        #expect(errored == [true])
        #expect(scripted.callCount == 2)
        guard case .finished = events.last else { Issue.record("expected finished last"); return }
    }

    @Test("iteration cap stops the loop with no extra generation")
    func iterationCap() async throws {
        let call = ToolCallRequest(id: "call_x", name: "loop_tool", arguments: [:])
        // More scripted turns than any cap under test; each keeps calling a tool.
        let scripted = ScriptedGenerate(Array(repeating: [toolCallChunk(call)], count: 20))
        let session = ToolCallingSession(
            generate: { scripted.generate($0) },
            callTool: { _, _, _ in "ok" },
            toolIndex: ["loop_tool": "srv"],
            maxIterations: 2
        )

        var events: [ToolLoopEvent] = []
        for try await event in session.run(request("go")) { events.append(event) }

        #expect(scripted.callCount == 2)  // exactly maxIterations generations
        guard case .finished = events.last else { Issue.record("expected finished last"); return }
    }

    @Test("a slow tool times out into an error result; loop continues")
    func toolTimeout() async throws {
        let call = ToolCallRequest(id: "call_1", name: "slow", arguments: [:])
        let scripted = ScriptedGenerate([
            [toolCallChunk(call)],
            [GenerateChunk(text: "moving on", finishReason: .stop)],
        ])
        let session = ToolCallingSession(
            generate: { scripted.generate($0) },
            callTool: { _, _, _ in
                try await Task.sleep(for: .seconds(10))
                return "never"
            },
            toolIndex: ["slow": "srv"],
            toolTimeout: .milliseconds(50)
        )

        var events: [ToolLoopEvent] = []
        for try await event in session.run(request("go")) { events.append(event) }

        let results = events.compactMap { event -> (String, Bool)? in
            if case .toolResult(_, let content, let isError) = event { return (content, isError) }
            return nil
        }
        #expect(results.count == 1)
        #expect(results[0].1 == true)
        #expect(results[0].0.contains("timed out"))
        #expect(scripted.callCount == 2)  // loop continued after the timeout
    }

    @Test("cancelling mid-tool unwinds the loop and runs no further generation")
    func cancellationMidTool() async throws {
        let call = ToolCallRequest(id: "call_1", name: "slow", arguments: [:])
        let scripted = ScriptedGenerate([
            [toolCallChunk(call)],
            [GenerateChunk(text: "should not run", finishReason: .stop)],
        ])
        let gate = CallGate()
        let session = ToolCallingSession(
            generate: { scripted.generate($0) },
            callTool: { _, _, _ in
                await gate.open()
                try await Task.sleep(for: .seconds(10))
                return "late"
            },
            toolIndex: ["slow": "srv"],
            toolTimeout: .seconds(60)
        )

        let stream = session.run(request("go"))
        let consumer = Task {
            // Drain the whole stream; cancellation (below) ends the iteration.
            for try await _ in stream {}
        }

        // Wait until callTool is provably running (gate opened just before its
        // sleep) BEFORE cancelling — otherwise a cancel that pre-empts callTool
        // would leave the gate closed and this await hanging.
        await gate.wait()
        consumer.cancel()              // cancel → onTermination → loop task cancelled
        _ = try? await consumer.value
        try? await Task.sleep(for: .milliseconds(50))  // let unwinding settle

        #expect(scripted.callCount == 1)  // the second generation never ran
    }

    // MARK: - Helpers

    private func request(_ prompt: String) -> GenerateRequest {
        GenerateRequest(model: "m", messages: [ChatMessage(role: .user, content: prompt)])
    }
}
