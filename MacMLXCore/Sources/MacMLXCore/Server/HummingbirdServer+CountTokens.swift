// Copyright © 2026 macMLX. English comments only.

import Foundation
import Hummingbird
import NIOCore

// MARK: - Anthropic token counting (`POST /v1/messages/count_tokens`)

extension HummingbirdServer {

    /// Anthropic `POST /v1/messages/count_tokens` → `{"input_tokens": N}`.
    ///
    /// The body is a `/v1/messages` body without the generation fields
    /// (`max_tokens` is optional and ignored, as is `stream`). It becomes the
    /// same `GenerateRequest` `/v1/messages` builds (`anthropicGenerateRequest`
    /// — turns, system placement, tools, `thinking`, images), and the engine
    /// counts the rendered prompt exactly as generation tokenizes it
    /// (`InferenceEngine.countPromptTokens`).
    ///
    /// Counting needs the model's tokenizer and template, so the model is
    /// resolved and loaded as for a generation, under the generation lock
    /// (`beginGeneration`): a count waits behind an in-flight generation,
    /// and may cold-swap the model. An unknown model is a 404, an engine that
    /// cannot count a 501, all in Anthropic's error envelope.
    func handleAnthropicCountTokens(
        request: Request,
        context: BasicRequestContext
    ) async throws -> Response {
        let buffer = try await request.body.collect(upTo: context.maxUploadSize)
        guard let data = buffer.getData(at: buffer.readerIndex, length: buffer.readableBytes) else {
            return anthropicErrorResponse(status: .badRequest, message: "Empty request body", type: "invalid_request_error")
        }
        let genRequest: GenerateRequest
        do {
            (genRequest, _) = try await anthropicGenerateRequest(from: data, requiresMaxTokens: false)
        } catch let error as AnthropicRequestError {
            return anthropicErrorResponse(status: .badRequest, message: error.message, type: "invalid_request_error")
        }

        let engine: any InferenceEngine
        do {
            engine = try await beginGeneration(genRequest.model)
        } catch let err as ModelSwapError {
            return anthropicSwapErrorResponse(err)
        } catch {
            return anthropicErrorResponse(
                status: .internalServerError, message: error.localizedDescription, type: "api_error")
        }
        defer { Task { [weak self] in await self?.releaseGenerationLock() } }

        let modelID = await engine.loadedModel?.id ?? genRequest.model
        await markInFlight(modelID, true)
        defer { Task { [weak self] in await self?.markInFlight(modelID, false) } }

        let inputTokens: Int
        do {
            inputTokens = try await engine.countPromptTokens(genRequest)
        } catch EngineError.tokenCountingUnsupported {
            return anthropicErrorResponse(
                status: .notImplemented,
                message: EngineError.tokenCountingUnsupported.localizedDescription,
                type: "api_error"
            )
        } catch {
            return anthropicErrorResponse(
                status: .internalServerError, message: error.localizedDescription, type: "api_error")
        }
        return try jsonResponseAny(["input_tokens": inputTokens])
    }
}
