// ToolCallingSession.swift
// MacMLXCore
//
// The chat-side MCP tool loop (v0.5). Drives a generation, detects the tool
// calls the model asked for, routes each to its MCP server, feeds the results
// back, and re-generates — until the model answers without calling a tool, an
// iteration cap is hit, or the consumer cancels.
//
// Dependencies are injected as closures so the whole loop is testable with
// zero MLX and zero real MCP: a stub `generate` scripts model turns and a stub
// `callTool` scripts tool results. A convenience initialiser wraps a real
// `MCPClientPool` (converting arguments via `ToolValueBridge` and extracting
// text from the `CallTool.Result`).
//
// `ToolLoopEvent` lives here rather than in its own file: it is the session's
// output vocabulary and has no meaning apart from it.

import Foundation
import MCP

// MARK: - Events

/// One event emitted while a ``ToolCallingSession`` runs.
public enum ToolLoopEvent: Sendable {
    /// A streamed chunk from the current model turn — forward verbatim to the
    /// UI. The terminal chunk of a turn may carry `toolCalls` +
    /// `finishReason == .toolCalls`; the loop acts on those internally.
    case assistantDelta(GenerateChunk)
    /// A tool call is about to be dispatched to `server`.
    case toolCallStarted(ToolCallRequest, server: String)
    /// A tool call finished (or failed). `content` is the text fed back to the
    /// model as the tool result; `isError` is true for unknown tool / throw /
    /// timeout, in which case `content` is a synthetic error message.
    case toolResult(id: String, content: String, isError: Bool)
    /// The loop finished. Carries the finish reason of the last model turn.
    case finished(FinishReason?)
}

// MARK: - Errors

/// Errors raised inside the tool loop's `callTool` path. Surfaced to the model
/// as tool-result text, never thrown out of the loop.
public enum ToolCallingError: Error, LocalizedError, Sendable {
    /// The per-call timeout elapsed before the tool returned.
    case timedOut(tool: String, seconds: Double)
    /// The tool ran but reported a failure (`CallTool.Result.isError == true`).
    case toolReportedError(String)

    public var errorDescription: String? {
        switch self {
        case .timedOut(let tool, let seconds):
            return "Tool '\(tool)' timed out after \(seconds)s"
        case .toolReportedError(let message):
            return message
        }
    }
}

// MARK: - Session

/// Runs the generate → call-tools → re-generate loop for one user turn.
///
/// Value type holding only immutable configuration and `@Sendable` closures,
/// so it crosses isolation boundaries freely.
public struct ToolCallingSession: Sendable {

    /// Starts a model generation for the given request.
    private let generate: @Sendable (GenerateRequest) -> AsyncThrowingStream<GenerateChunk, Error>
    /// Calls one tool: `(server, toolName, arguments) -> resultText`. Throws on
    /// failure (dead server, transport error, tool-reported error); the loop
    /// converts a throw into an error tool-result and keeps going.
    private let callTool: @Sendable (String, String, [String: JSONValue]?) async throws -> String
    /// Tool name → owning MCP server name. A call whose name is absent here is
    /// answered with a synthetic "unknown tool" error result.
    private let toolIndex: [String: String]
    /// Hard cap on generate→tools iterations, so a model that keeps calling
    /// tools can't loop forever.
    private let maxIterations: Int
    /// Per-call wall-clock budget. `callTool` has no timeout of its own
    /// (`MCPClientPool.callTool`), so the loop enforces one.
    private let toolTimeout: Duration

    /// Designated initialiser — inject `generate` and `callTool` directly.
    /// Tests use this with stubs; no `MCPClientPool` required.
    public init(
        generate: @escaping @Sendable (GenerateRequest) -> AsyncThrowingStream<GenerateChunk, Error>,
        callTool: @escaping @Sendable (String, String, [String: JSONValue]?) async throws -> String,
        toolIndex: [String: String],
        maxIterations: Int = 6,
        toolTimeout: Duration = .seconds(60)
    ) {
        self.generate = generate
        self.callTool = callTool
        self.toolIndex = toolIndex
        self.maxIterations = maxIterations
        self.toolTimeout = toolTimeout
    }

    /// Convenience initialiser wrapping a live ``MCPClientPool``. Converts
    /// macMLX `JSONValue` arguments to `MCP.Value`, extracts text content from
    /// the result, and throws ``ToolCallingError/toolReportedError(_:)`` when
    /// the server flags `isError`, so the loop reports it back to the model.
    public init(
        generate: @escaping @Sendable (GenerateRequest) -> AsyncThrowingStream<GenerateChunk, Error>,
        pool: MCPClientPool,
        toolIndex: [String: String],
        maxIterations: Int = 6,
        toolTimeout: Duration = .seconds(60)
    ) {
        self.init(
            generate: generate,
            callTool: { server, name, arguments in
                let mcpArguments = arguments?.mapValues { ToolValueBridge.mcpValue(from: $0) }
                let result = try await pool.callTool(
                    server: server, name: name, arguments: mcpArguments)
                let text = Self.text(from: result)
                if result.isError == true {
                    throw ToolCallingError.toolReportedError(
                        text.isEmpty ? "The tool reported an error." : text)
                }
                return text
            },
            toolIndex: toolIndex,
            maxIterations: maxIterations,
            toolTimeout: toolTimeout
        )
    }

    /// Run the loop for `request`, streaming ``ToolLoopEvent``s.
    ///
    /// Cancelling iteration of the returned stream propagates into the running
    /// generation and any in-flight tool call (mirrors `MLXSwiftEngine.generate`'s
    /// `onTermination` hook), so a Stop button unwinds the whole loop.
    public func run(_ request: GenerateRequest) -> AsyncThrowingStream<ToolLoopEvent, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    try await runLoop(request, into: continuation)
                    continuation.finish()
                } catch is CancellationError {
                    continuation.finish(throwing: CancellationError())
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { @Sendable _ in
                task.cancel()
            }
        }
    }

    // MARK: - Loop

    private func runLoop(
        _ request: GenerateRequest,
        into continuation: AsyncThrowingStream<ToolLoopEvent, Error>.Continuation
    ) async throws {
        // The evolving conversation. The system prompt stays on the request
        // (applied by the engine via `allMessages`); we only grow `messages`.
        var workingMessages = request.messages
        var iteration = 0

        while true {
            try Task.checkCancellation()

            let turnRequest = GenerateRequest(
                model: request.model,
                messages: workingMessages,
                systemPrompt: request.systemPrompt,
                parameters: request.parameters,
                templateKwargs: request.templateKwargs,
                tools: request.tools
            )

            var assistantText = ""
            var toolCalls: [ToolCallRequest] = []
            var finishReason: FinishReason?

            for try await chunk in generate(turnRequest) {
                try Task.checkCancellation()
                assistantText += chunk.text
                if let calls = chunk.toolCalls, !calls.isEmpty {
                    toolCalls = calls
                }
                if let reason = chunk.finishReason {
                    finishReason = reason
                }
                continuation.yield(.assistantDelta(chunk))
            }

            // No tool calls → the model answered. Done.
            if toolCalls.isEmpty {
                continuation.yield(.finished(finishReason))
                return
            }

            iteration += 1

            // Record the assistant's tool-call turn, then one tool-result turn
            // per call, so the next generation sees the full exchange.
            workingMessages.append(
                ChatMessage(role: .assistant, content: assistantText, toolCalls: toolCalls))

            for call in toolCalls {
                try Task.checkCancellation()
                let content: String
                let isError: Bool
                if let server = toolIndex[call.name] {
                    continuation.yield(.toolCallStarted(call, server: server))
                    (content, isError) = try await runTool(call, server: server)
                } else {
                    // Unknown tool → synthetic error result fed back to the
                    // model so it can recover, never a crash.
                    content = "Error: no MCP server provides a tool named '\(call.name)'."
                    isError = true
                }
                continuation.yield(.toolResult(id: call.id, content: content, isError: isError))
                workingMessages.append(
                    ChatMessage(role: .tool, content: content, toolCallID: call.id))
            }

            // Iteration cap: stop after running this turn's tools — no bonus
            // final generation pass.
            if iteration >= maxIterations {
                continuation.yield(.finished(finishReason))
                return
            }
        }
    }

    /// Execute one known tool call, returning `(resultText, isError)`. Wraps
    /// `callTool` in the cancellation-aware timeout race; a timeout or throw
    /// becomes an error result. Rethrows only genuine consumer cancellation,
    /// which must unwind the whole loop.
    private func runTool(
        _ call: ToolCallRequest,
        server: String
    ) async throws -> (content: String, isError: Bool) {
        do {
            let content = try await Self.withTimeout(toolTimeout, tool: call.name) {
                try await callTool(server, call.name, call.arguments)
            }
            return (content, false)
        } catch is CancellationError {
            throw CancellationError()
        } catch {
            return (
                "Error: tool '\(call.name)' failed: \(error.localizedDescription)",
                true
            )
        }
    }

    // MARK: - Helpers

    /// Race `operation` against a timeout. On timeout the operation task is
    /// cancelled and ``ToolCallingError/timedOut(tool:seconds:)`` is thrown.
    /// External cancellation propagates as `CancellationError`.
    static func withTimeout<T: Sendable>(
        _ duration: Duration,
        tool: String,
        operation: @escaping @Sendable () async throws -> T
    ) async throws -> T {
        try await withThrowingTaskGroup(of: T.self) { group in
            group.addTask { try await operation() }
            group.addTask {
                try await Task.sleep(for: duration)
                throw ToolCallingError.timedOut(
                    tool: tool,
                    seconds: Double(duration.components.seconds)
                )
            }
            defer { group.cancelAll() }
            guard let first = try await group.next() else {
                throw CancellationError()
            }
            return first
        }
    }

    /// Join the text of every `.text` content block in a tool result.
    static func text(from result: CallTool.Result) -> String {
        result.content
            .compactMap { content -> String? in
                if case .text(let text, _, _) = content { return text }
                return nil
            }
            .joined(separator: "\n")
    }
}
