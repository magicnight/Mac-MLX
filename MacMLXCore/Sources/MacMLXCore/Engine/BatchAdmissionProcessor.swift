// Copyright © 2026 macMLX. English comments only.

import MLXLMCommon

/// Serial, off-mutex admission tokenizer for the engine ``BatchGenerationServing``
/// seam (A2d-2 H2). Owns a DEDICATED tokenizer instance — loaded once at model load,
/// SEPARATE from the `ModelContext.tokenizer` the drive loop detokenizes with — plus
/// the resident model's `MessageGenerator`, and renders a request's chat template to
/// prompt token ids.
///
/// ## Why an actor, and why a separate tokenizer
/// Admission tokenization must run OFF the container's serial-access mutex — the
/// drive loop holds that mutex for whole cohorts (bounded by the step budget, but
/// still long), so routing admission through it would serialize every new client
/// behind the running cohort (see the seam's type doc). Running admission off-mutex,
/// however, is exactly what made the OLD code racy: it reused the container's ONE
/// `processor`, whose tokenizer IS `context.tokenizer`, so two hazards existed —
///   1. concurrent `submit`s ran `applyChatTemplate` on the same non-`Sendable`
///      tokenizer at once, and
///   2. any such admission raced the drive loop's `context.tokenizer.decode` running
///      inside the mutex.
/// This actor closes both: (1) actor isolation serializes concurrent admissions, and
/// (2) the tokenizer here is a DISTINCT object from `context.tokenizer`, so the drive
/// loop's detokenizer can never touch the same instance — the two domains are fully
/// isolated even though neither takes the other's lock.
///
/// ## Why load a second tokenizer instead of one per request
/// A tokenizer is a multi-MB parse (`AutoTokenizer.from(modelFolder:)`), so building
/// one PER admission would add that cost to every turn under concurrency. Loading it
/// ONCE at model load and serializing keeps admission cheap while still isolated —
/// only the tokenizer is duplicated, never the multi-GB model weights. The two
/// instances load from the SAME model directory, so their chat template and vocab are
/// identical: batched-path prompt tokens match the legacy path byte-for-byte.
actor BatchAdmissionProcessor {
    private let tokenizer: any Tokenizer
    private let messageGenerator: any MessageGenerator

    init(tokenizer: any Tokenizer, messageGenerator: any MessageGenerator) {
        self.tokenizer = tokenizer
        self.messageGenerator = messageGenerator
    }

    /// Render `input`'s chat template to prompt token ids, mirroring upstream
    /// `LLMUserInputProcessor.prepare` exactly (message generation →
    /// `applyChatTemplate`, with the same `missingChatTemplate` → plain-text-encode
    /// fallback) but returning ids directly — the batched path never needs the
    /// `MLXArray`.
    ///
    /// `sending`: `UserInput` is not `Sendable` (it can hold `CIImage`/`AVAsset`), so
    /// the caller transfers its freshly-built, un-aliased value into this actor. The
    /// whole render then runs as ONE serialized actor step, so no two admissions —
    /// and nothing on the drive loop — touch this tokenizer concurrently.
    func prepare(_ input: sending UserInput) throws -> [Int] {
        let messages = messageGenerator.generate(from: input)
        do {
            return try tokenizer.applyChatTemplate(
                messages: messages, tools: input.tools, additionalContext: input.additionalContext)
        } catch MLXLMCommon.TokenizerError.missingChatTemplate {
            let prompt =
                messages
                .compactMap { $0["content"] as? String }
                .joined(separator: "\n\n")
            return tokenizer.encode(text: prompt)
        }
    }
}
