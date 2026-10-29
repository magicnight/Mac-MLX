// Copyright © 2026 macMLX. English comments only.

import Foundation
import HTTPTypes
import Hummingbird
import NIOCore

// MARK: - OpenAI Responses API (`POST /v1/responses`)
//
// A stateless translation layer over the chat path: the request becomes the
// same `GenerateRequest` `/v1/chat/completions` builds
// (`ResponsesInputTranslator`), it is generated on the single-stream path
// (`beginGeneration` → engine → `nextGenerationStep`), and the chunk stream is
// rendered as Responses output items and events (`ResponsesOutputBuilder`,
// `ResponsesEventStream`). No response is stored. The batched path is never
// consulted: agent clients always send tools, which its responders do not
// parse.

extension HummingbirdServer {

    /// Ceiling on a `/v1/responses` body. A Responses client is stateless
    /// toward macMLX and replays the whole conversation every turn — tool
    /// outputs, base64 screenshots and all — so the route-wide 2 MB
    /// `context.maxUploadSize` would turn a long agent session into a
    /// permanent 413 a few turns in.
    static let responsesBodyLimitBytes = 64 * 1024 * 1024

    /// The output budget when a request names none. Codex never sends
    /// `max_output_tokens`, and the chat path's 2048 is spent by a thinking
    /// model's reasoning before it reaches the tool call, which would end
    /// every such turn `incomplete` with no call in it.
    static let responsesDefaultMaxOutputTokens = 32_768

    func handleResponses(
        request: Request,
        context: BasicRequestContext
    ) async throws -> Response {
        let buffer: ByteBuffer
        do {
            buffer = try await request.body.collect(upTo: Self.responsesBodyLimitBytes)
        } catch {
            // Only the ceiling is a 413 (see `isUploadTooLarge`); anything else
            // takes Hummingbird's own error path, as on the other routes.
            guard Self.isUploadTooLarge(error) else { throw error }
            return errorResponse(
                status: .contentTooLarge,
                message: "Request body exceeds the \(Self.responsesBodyLimitBytes)-byte limit.",
                code: "request_too_large"
            )
        }
        guard let data = buffer.getData(at: buffer.readerIndex, length: buffer.readableBytes) else {
            return errorResponse(status: .badRequest, message: "Empty request body", code: "invalid_request_error")
        }
        let responsesRequest: ResponsesRequest
        do {
            responsesRequest = try JSONDecoder().decode(ResponsesRequest.self, from: data)
        } catch {
            return errorResponse(
                status: .badRequest,
                message: "Invalid JSON: \(error.localizedDescription)",
                code: "invalid_request_error"
            )
        }
        let translation: ResponsesInputTranslator.Translation
        do {
            translation = try ResponsesInputTranslator.translate(responsesRequest)
        } catch let error as ResponsesInputTranslator.TranslationError {
            return errorResponse(status: .badRequest, message: error.message, code: "invalid_request_error")
        }
        if !translation.droppedToolTypes.isEmpty {
            await LogManager.shared.debug(
                "/v1/responses: not offering tools of type "
                    + translation.droppedToolTypes.joined(separator: ", ")
                    + " (hosted tools run on OpenAI's side; macMLX has none)",
                category: .http)
        }

        // The translator has already rejected a missing or empty `model`.
        let model = responsesRequest.model ?? ""
        let parameters = GenerationParameters(
            temperature: responsesRequest.temperature ?? 0.7,
            topP: responsesRequest.top_p ?? 0.95,
            maxTokens: responsesRequest.max_output_tokens ?? Self.responsesDefaultMaxOutputTokens,
            stream: responsesRequest.stream ?? false
        )
        // `reasoning.effort` "none"/"minimal" switches a thinking template
        // (Qwen3's `enable_thinking`) off over the per-model setting; any other
        // effort leaves the model's default.
        var configuredTemplateKwargs = await templateKwargs(for: model)
        if translation.disablesThinking {
            var kwargs = configuredTemplateKwargs ?? [:]
            kwargs["enable_thinking"] = .bool(false)
            configuredTemplateKwargs = kwargs
        }
        let genRequest = GenerateRequest(
            model: model,
            messages: translation.messages,
            systemPrompt: translation.systemPrompt,
            parameters: parameters,
            templateKwargs: Self.effectiveTemplateKwargs(
                configuredTemplateKwargs, constrainedBy: translation.responseFormat),
            tools: translation.tools,
            responseFormat: translation.responseFormat
        )
        let envelope = ResponsesOutputBuilder.Envelope(
            id: ResponsesOutputBuilder.newID("resp"),
            createdAt: Int(Date().timeIntervalSince1970),
            request: responsesRequest,
            parameters: parameters
        )
        let exposeCacheFigures = Self.exposeCacheFigures(request)
        if parameters.stream {
            return await streamingResponsesResponse(
                genRequest: genRequest, envelope: envelope,
                toolNames: translation.toolNames, exposeCacheFigures: exposeCacheFigures)
        }
        return try await nonStreamingResponsesResponse(
            genRequest: genRequest, envelope: envelope,
            toolNames: translation.toolNames, exposeCacheFigures: exposeCacheFigures)
    }

    /// The whole response as one JSON body. A generation that fails after it
    /// started (an engine error or a stall) is a `failed` response object, not
    /// an HTTP error: it is the same outcome the stream reports in-band.
    private func nonStreamingResponsesResponse(
        genRequest: GenerateRequest,
        envelope: ResponsesOutputBuilder.Envelope,
        toolNames: ResponsesToolNameMap,
        exposeCacheFigures: Bool
    ) async throws -> Response {
        let engine: any InferenceEngine
        do {
            engine = try await beginGeneration(genRequest.model)
        } catch let err as ModelSwapError {
            return openAIStyleSwapErrorResponse(err)
        } catch {
            return errorResponse(status: .internalServerError, message: error.localizedDescription, code: "load_failed")
        }
        defer { Task { [weak self] in await self?.releaseGenerationLock() } }

        let modelID = await engine.loadedModel?.id ?? genRequest.model
        await markInFlight(modelID, true)
        defer { Task { [weak self] in await self?.markInFlight(modelID, false) } }

        // A constrained answer is JSON from its first byte, and a think tag in
        // a string value is data, so the splitter is bypassed (as on the chat
        // path).
        let splitsReasoning = genRequest.responseFormat == nil
        let startInReasoning = splitsReasoning ? await engine.promptOpensThinkBlock(genRequest) : false
        var splitter = ReasoningStreamSplitter(startInReasoning: startInReasoning)
        let box = ChunkIteratorBox(engine.generate(genRequest))
        var reasoning = ""
        var answer = ""
        var usage: TokenUsage?
        var finishReason: FinishReason?
        var toolCalls: [ToolCallRequest] = []
        var failure: String?

        do {
            loop: while true {
                switch try await nextGenerationStep(box, stallTimeout: stallTimeoutSeconds) {
                case .finished:
                    break loop
                case .stalled:
                    failure = "Generation stalled: no output for over \(Int(stallTimeoutSeconds))s."
                    break loop
                case .chunk(let chunk):
                    if let chunkUsage = chunk.usage { usage = chunkUsage }
                    if let calls = chunk.toolCalls, !calls.isEmpty {
                        toolCalls = calls.map(ResponsesOutputBuilder.withCallID)
                    }
                    let (r, a) = splitsReasoning ? splitter.push(chunk.text) : ("", chunk.text)
                    reasoning += r
                    answer += a
                    if let reason = chunk.finishReason {
                        finishReason = reason
                        let (rTail, aTail) = splitsReasoning ? splitter.finish() : ("", "")
                        reasoning += rTail
                        answer += aTail
                    }
                }
            }
        } catch {
            failure = error.localizedDescription
        }
        incrementTokens(usage?.completionTokens ?? 0)
        // As in the stream: the answer starts at its first non-whitespace text.
        answer = String(answer.drop(while: \.isWhitespace))

        var output: [[String: Any]] = []
        if !reasoning.isEmpty {
            output.append(ResponsesOutputBuilder.reasoningItem(id: ResponsesOutputBuilder.newID("rs"), text: reasoning))
        }
        let outcome: ResponsesOutputBuilder.Outcome = failure.map { .failed($0, code: "server_error") }
            ?? .finished(finishReason)
        if !answer.isEmpty || (toolCalls.isEmpty && failure == nil) {
            output.append(ResponsesOutputBuilder.messageItem(
                id: ResponsesOutputBuilder.newID("msg"), text: answer,
                status: ResponsesOutputBuilder.itemStatus(outcome)))
        }
        for call in toolCalls {
            let prefix = ResponsesOutputBuilder.isCustomToolCall(call, toolNames: toolNames) ? "ctc" : "fc"
            output.append(ResponsesOutputBuilder.toolCallItem(
                id: ResponsesOutputBuilder.newID(prefix), call: call, toolNames: toolNames, done: true))
        }

        let (status, incompleteDetails, error) = ResponsesOutputBuilder.statusFields(outcome)
        var body = ResponsesOutputBuilder.responseObject(
            envelope,
            status: status,
            output: output,
            usage: ResponsesOutputBuilder.usageObject(usage, exposeCacheFigures: exposeCacheFigures),
            incompleteDetails: incompleteDetails,
            error: error
        )
        // The SDKs' convenience accessor, sent in the non-streaming body only.
        body["output_text"] = answer
        return try jsonResponseAny(body)
    }

    /// The response as Responses SSE events. `response.created` and
    /// `response.in_progress` go out before the generation lock is taken, so
    /// a client waiting behind another generation (or a cold swap) sees the
    /// response start. Reasoning and text stream as their own items while
    /// chunks arrive — reasoning as both its summary part and its
    /// `reasoning_text` content (see `ResponsesOutputBuilder.reasoningItem`).
    /// Tool calls, which the engine reports on the terminal chunk, follow as
    /// whole items, and exactly one terminal event ends the stream. An unknown model is a 404 before any byte is sent; a failure
    /// once the stream is open is `response.failed`.
    private func streamingResponsesResponse(
        genRequest: GenerateRequest,
        envelope: ResponsesOutputBuilder.Envelope,
        toolNames: ResponsesToolNameMap,
        exposeCacheFigures: Bool
    ) async -> Response {
        // A1: reject an unknown model with a real 404 before headers are sent;
        // the locked resolve and load still happen inside the body (SRV-2).
        if await !canResolveModel(genRequest.model) {
            return openAIStyleSwapErrorResponse(.modelNotFound(id: genRequest.model))
        }
        let server = self

        let responseBody = ResponseBody { writer in
            var events = ResponsesEventStream()
            // Frames encoded since the last flush.
            var frames: [String] = []
            // The finished items, in `output_index` order.
            var output: [[String: Any]] = []
            // The item text is streaming into: reasoning (true) or the
            // message (false); nil when none is open.
            var openIsReasoning: Bool?
            var openID = ""
            var openText = ""
            var messageEmitted = false

            func emit(_ type: String, _ payload: [String: Any]) throws {
                frames.append(try events.frame(type, payload))
            }
            func openItem(reasoning: Bool) throws {
                openIsReasoning = reasoning
                openID = ResponsesOutputBuilder.newID(reasoning ? "rs" : "msg")
                openText = ""
                let index = output.count
                try emit("response.output_item.added", [
                    "output_index": index,
                    "item": reasoning
                        ? ResponsesOutputBuilder.reasoningItem(id: openID, text: nil)
                        : ResponsesOutputBuilder.messageItem(id: openID, text: nil, status: "in_progress"),
                ])
                if reasoning {
                    try emit("response.reasoning_summary_part.added", [
                        "item_id": openID, "output_index": index, "summary_index": 0,
                        "part": ResponsesOutputBuilder.summaryTextPart(""),
                    ])
                }
                try emit("response.content_part.added", [
                    "item_id": openID, "output_index": index, "content_index": 0,
                    "part": reasoning
                        ? ResponsesOutputBuilder.reasoningTextPart("")
                        : ResponsesOutputBuilder.outputTextPart(""),
                ])
            }
            func closeItem(status: String = "completed") throws {
                guard let reasoning = openIsReasoning else { return }
                let index = output.count
                let item: [String: Any]
                if reasoning {
                    try emit("response.reasoning_summary_text.done", [
                        "item_id": openID, "output_index": index, "summary_index": 0, "text": openText,
                    ])
                    try emit("response.reasoning_summary_part.done", [
                        "item_id": openID, "output_index": index, "summary_index": 0,
                        "part": ResponsesOutputBuilder.summaryTextPart(openText),
                    ])
                    try emit("response.reasoning_text.done", [
                        "item_id": openID, "output_index": index, "content_index": 0, "text": openText,
                    ])
                    try emit("response.content_part.done", [
                        "item_id": openID, "output_index": index, "content_index": 0,
                        "part": ResponsesOutputBuilder.reasoningTextPart(openText),
                    ])
                    item = ResponsesOutputBuilder.reasoningItem(id: openID, text: openText)
                } else {
                    try emit("response.output_text.done", [
                        "item_id": openID, "output_index": index, "content_index": 0,
                        "text": openText, "logprobs": [] as [Any],
                    ])
                    try emit("response.content_part.done", [
                        "item_id": openID, "output_index": index, "content_index": 0,
                        "part": ResponsesOutputBuilder.outputTextPart(openText),
                    ])
                    item = ResponsesOutputBuilder.messageItem(id: openID, text: openText, status: status)
                    messageEmitted = true
                }
                try emit("response.output_item.done", ["output_index": index, "item": item])
                output.append(item)
                openIsReasoning = nil
            }
            // Text of one kind continues the open item, or closes it and opens
            // a new one — normally one reasoning item, then the message. A
            // message opens on its first non-whitespace text: the blank lines
            // a thinking model emits after `</think>` are not an answer, and a
            // tool-only turn must not gain a whitespace message item.
            func stream(_ text: String, reasoning: Bool) throws {
                var text = text
                if !reasoning, openIsReasoning != false {
                    text = String(text.drop(while: \.isWhitespace))
                }
                guard !text.isEmpty else { return }
                if openIsReasoning != reasoning {
                    try closeItem()
                    try openItem(reasoning: reasoning)
                }
                openText += text
                let index = output.count
                if reasoning {
                    try emit("response.reasoning_summary_text.delta", [
                        "item_id": openID, "output_index": index, "summary_index": 0, "delta": text,
                    ])
                    try emit("response.reasoning_text.delta", [
                        "item_id": openID, "output_index": index, "content_index": 0, "delta": text,
                    ])
                } else {
                    try emit("response.output_text.delta", [
                        "item_id": openID, "output_index": index, "content_index": 0, "delta": text,
                        "logprobs": [] as [Any],
                    ])
                }
            }
            func emitToolCall(_ call: ToolCallRequest) throws {
                let isCustom = ResponsesOutputBuilder.isCustomToolCall(call, toolNames: toolNames)
                let id = ResponsesOutputBuilder.newID(isCustom ? "ctc" : "fc")
                let index = output.count
                let payload = ResponsesOutputBuilder.toolCallPayload(call, toolNames: toolNames)
                try emit("response.output_item.added", [
                    "output_index": index,
                    "item": ResponsesOutputBuilder.toolCallItem(id: id, call: call, toolNames: toolNames, done: false),
                ])
                if isCustom {
                    try emit("response.custom_tool_call_input.delta", [
                        "item_id": id, "output_index": index, "delta": payload,
                    ])
                    try emit("response.custom_tool_call_input.done", [
                        "item_id": id, "output_index": index, "input": payload,
                    ])
                } else {
                    try emit("response.function_call_arguments.delta", [
                        "item_id": id, "output_index": index, "delta": payload,
                    ])
                    try emit("response.function_call_arguments.done", [
                        "item_id": id, "output_index": index, "arguments": payload,
                    ])
                }
                let item = ResponsesOutputBuilder.toolCallItem(id: id, call: call, toolNames: toolNames, done: true)
                try emit("response.output_item.done", ["output_index": index, "item": item])
                output.append(item)
            }

            try emit("response.created", [
                "response": ResponsesOutputBuilder.responseObject(
                    envelope, status: "in_progress", output: [], usage: nil),
            ])
            try emit("response.in_progress", [
                "response": ResponsesOutputBuilder.responseObject(
                    envelope, status: "in_progress", output: [], usage: nil),
            ])
            for frame in frames { try await writer.write(ByteBuffer(string: frame)) }
            frames.removeAll()

            var usage: TokenUsage?
            var finishReason: FinishReason?
            var toolCalls: [ToolCallRequest] = []
            var failure: String?
            var failureCode = "server_error"

            // Lock-scope invariant (SRV-2/SRV-3): acquire, cold-swap and
            // generate all happen inside this closure, released by `defer`.
            var engine: (any InferenceEngine)?
            do {
                engine = try await server.beginGeneration(genRequest.model)
            } catch ModelSwapError.modelNotFound(let id) {
                // The codes `openAIStyleSwapErrorResponse` gives these on a
                // response that has not started.
                failure = "Model not found: \(id)."
                failureCode = "model_not_found"
            } catch ModelSwapError.loadFailed(let id, let reason) {
                failure = "Failed to load \(id): \(reason)"
                failureCode = "load_failed"
            } catch {
                failure = error.localizedDescription
            }
            if let engine {
                defer { Task { await server.releaseGenerationLock() } }
                let modelID = await engine.loadedModel?.id ?? genRequest.model
                await server.markInFlight(modelID, true)
                defer { Task { await server.markInFlight(modelID, false) } }

                let splitsReasoning = genRequest.responseFormat == nil
                let startInReasoning = splitsReasoning ? await engine.promptOpensThinkBlock(genRequest) : false
                var splitter = ReasoningStreamSplitter(startInReasoning: startInReasoning)
                let box = ChunkIteratorBox(engine.generate(genRequest))
                let stallTimeout = server.stallTimeoutSeconds
                do {
                    loop: while true {
                        switch try await nextGenerationStep(box, stallTimeout: stallTimeout) {
                        case .finished:
                            break loop
                        case .stalled:
                            failure = "Generation stalled: no output for over \(Int(stallTimeout))s."
                            break loop
                        case .chunk(let chunk):
                            if let chunkUsage = chunk.usage { usage = chunkUsage }
                            if let calls = chunk.toolCalls, !calls.isEmpty {
                                toolCalls = calls.map(ResponsesOutputBuilder.withCallID)
                            }
                            var (r, a) = splitsReasoning ? splitter.push(chunk.text) : ("", chunk.text)
                            if let reason = chunk.finishReason {
                                finishReason = reason
                                let (rTail, aTail) = splitsReasoning ? splitter.finish() : ("", "")
                                r += rTail
                                a += aTail
                            }
                            try stream(r, reasoning: true)
                            try stream(a, reasoning: false)
                            for frame in frames { try await writer.write(ByteBuffer(string: frame)) }
                            frames.removeAll()
                        }
                    }
                } catch {
                    failure = error.localizedDescription
                }
                await server.incrementTokens(usage?.completionTokens ?? 0)
            }

            let outcome: ResponsesOutputBuilder.Outcome = failure.map { .failed($0, code: failureCode) }
                ?? .finished(finishReason)
            try closeItem(status: ResponsesOutputBuilder.itemStatus(outcome))
            for call in toolCalls {
                try emitToolCall(call)
            }
            // A finished response always has its message item, empty or not,
            // unless the model only called tools.
            if failure == nil, !messageEmitted, toolCalls.isEmpty {
                try openItem(reasoning: false)
                try closeItem(status: ResponsesOutputBuilder.itemStatus(outcome))
            }
            let (status, incompleteDetails, error) = ResponsesOutputBuilder.statusFields(outcome)
            let terminalEvent: String
            switch status {
            case "completed": terminalEvent = "response.completed"
            case "incomplete": terminalEvent = "response.incomplete"
            default: terminalEvent = "response.failed"
            }
            try emit(terminalEvent, [
                "response": ResponsesOutputBuilder.responseObject(
                    envelope,
                    status: status,
                    output: output,
                    usage: ResponsesOutputBuilder.usageObject(usage, exposeCacheFigures: exposeCacheFigures),
                    incompleteDetails: incompleteDetails,
                    error: error
                ),
            ])
            for frame in frames { try await writer.write(ByteBuffer(string: frame)) }
            try await writer.finish(nil)
        }

        var headers = HTTPFields()
        headers[.contentType] = "text/event-stream"
        headers[.cacheControl] = "no-cache"
        headers[.connection] = "keep-alive"
        return Response(status: .ok, headers: headers, body: responseBody)
    }
}
