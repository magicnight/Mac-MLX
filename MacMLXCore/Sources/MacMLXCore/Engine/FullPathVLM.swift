// Copyright © 2026 macMLX. English comments only.

import Foundation
import MLX
import MLXLMCommon
import MLXVLM

/// A vision-language load served on the text generation path
/// (`runLLMGeneration`) instead of mlx-swift-lm's stock VLM decoder, so its
/// requests keep the prompt cache, tool-call parsing, constraints and
/// replayed reasoning (design:
/// `docs/superpowers/specs/2026-10-31-gemma4-qwen35-vision-design.md`).
///
/// The engine decides it at load, from the checkpoint's `config.json`, for a
/// model the VLM factory loaded. Only a family in ``servedFamilies`` is served
/// this way, and a family joins that set only after its parity gate passed;
/// until then a VLM of that family keeps the stock path. `forced` is a test seam that
/// reaches the path for a family not in the set yet.
struct FullPathVLM: Sendable, Equatable {

    /// The `model_type`s this path knows how to drive: the message generator
    /// their processor renders with and the state a warm continuation needs.
    enum Family: String, Sendable, CaseIterable {
        case gemma4
        case qwen35 = "qwen3_5"

        /// The `processor_class` the VLM factory builds this family's
        /// processor from, as its checkpoints declare it. Upstream builds
        /// whatever class a checkpoint names, so a checkpoint naming another
        /// one would get another family's processor (see
        /// ``VisionCheckpoint``).
        var processorClass: String {
            switch self {
            case .gemma4: "Gemma4Processor"
            case .qwen35: "Qwen3VLProcessor"
            }
        }

        /// Whether a checkpoint with this `config.json` has the shape this
        /// family's parity run covered; another stays text until it passes
        /// its own run. Gemma 4 ran on E4B (`gemma-4-e4b-it-8bit`), whose
        /// text config sets none of `attention_k_eq_v`, bidirectional vision
        /// attention or the MoE block. The 31B dense checkpoint sets the first
        /// two, the 26B-A4B the third, and each runs in separate code in the
        /// text and the vision class. Read from `text_config`, else the root.
        func parityRunCovers(_ config: [String: Any]) -> Bool {
            switch self {
            case .gemma4:
                let text = config["text_config"] as? [String: Any] ?? config
                return text["attention_k_eq_v"] as? Bool != true
                    && text["use_bidirectional_attention"] as? String != "vision"
                    && text["enable_moe_block"] as? Bool != true
            case .qwen35:
                return false
            }
        }

        /// Tensor-name prefixes the vision tower's weights carry in an
        /// mlx-community conversion, after a leading `model.` is dropped.
        /// A checkpoint is complete only with every one present. Gemma 4
        /// needs the encoder and its projection into the text model; the
        /// audio tower is not needed, since upstream's `gemma4` sanitizer
        /// drops it. Qwen3.5's raw Hugging Face layout (`model.visual.*`) is
        /// not read yet; that family is not served.
        var visionWeightPrefixes: [String] {
            switch self {
            case .gemma4: ["vision_tower.", "embed_vision."]
            case .qwen35: ["vision_tower."]
            }
        }
    }

    /// Families whose checkpoints load through the VLM factory and run on the
    /// text generation path, each only for a checkpoint that carries its
    /// vision tower (``VisionCheckpoint``). A family joins once its parity
    /// gate passed: Gemma 4 on `gemma-4-e4b-it-8bit`. Qwen3.5/3.6 has not.
    static let servedFamilies: Set<Family> = [.gemma4]

    let family: Family

    /// The token ids where a media block can begin: the family's opening
    /// token (Qwen3.5's `<|vision_start|>`, Gemma 4's `boi`) and its
    /// placeholders. KV before the first of them does not depend on pixels.
    let mediaTokenIDs: Set<Int>

    /// Whether an image request may reuse the cache up to its first media
    /// block. Off for a Gemma 4 checkpoint with bidirectional vision
    /// attention: on a warm cache, upstream's overlay sees a mask wider than
    /// the remainder's token types and falls back to causal attention inside
    /// the image block, without an error.
    let reusesPrefixBeforeMedia: Bool

    init(family: Family, mediaTokenIDs: Set<Int>? = nil, reusesPrefixBeforeMedia: Bool = true) {
        self.family = family
        self.mediaTokenIDs = mediaTokenIDs ?? Self.mediaTokenIDs(family: family, config: [:])
        self.reusesPrefixBeforeMedia = reusesPrefixBeforeMedia
    }

    /// The full path for the checkpoint whose `config.json` is at
    /// `configURL`, or nil when it stays on the stock VLM path: an unreadable
    /// config, a `model_type` this path does not know, or (unless `forced`) a
    /// family not yet in ``servedFamilies``.
    static func resolve(configURL: URL, forced: Bool) -> FullPathVLM? {
        guard let data = try? Data(contentsOf: configURL),
            let root = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
            let modelType = root["model_type"] as? String,
            let family = Family(rawValue: modelType.lowercased())
        else { return nil }
        guard forced || servedFamilies.contains(family) else { return nil }
        let textConfig = root["text_config"] as? [String: Any]
        return FullPathVLM(
            family: family,
            mediaTokenIDs: mediaTokenIDs(family: family, config: root),
            reusesPrefixBeforeMedia: !(family == .gemma4
                && textConfig?["use_bidirectional_attention"] as? String == "vision"))
    }

    /// The media token ids `config` names, read with the keys and defaults
    /// upstream's configuration types use (`Qwen35Configuration`,
    /// `Gemma4Configuration`), so the cap lands where the model finds the
    /// block. Qwen3.6 checkpoints set `vision_start_token_id` to 248053;
    /// upstream's default, 248045, is `<|im_start|>`, which opens every Qwen
    /// chat, so a config without the key runs image requests cold.
    private static func mediaTokenIDs(family: Family, config: [String: Any]) -> Set<Int> {
        func id(_ key: String) -> Int? { config[key] as? Int }
        switch family {
        case .qwen35:
            let image = id("image_token_id") ?? 248_056
            let video = id("video_token_id") ?? 248_057
            return [
                id("vision_start_token_id") ?? 248_045, image, video,
                id("image_token_index") ?? image, id("video_token_index") ?? video,
            ]
        case .gemma4:
            return Set(
                [id("boi_token_id") ?? 255_999, id("image_token_id") ?? 258_880]
                    + [id("video_token_id"), id("audio_token_id")].compactMap { $0 })
        }
    }

    /// The key an image request looks the prompt cache up with, or nil when
    /// it runs cold: no media token in `tokens`, one at the very start, or
    /// ``reusesPrefixBeforeMedia`` off. It is the prefix before the first
    /// media token plus that token. The store always leaves a key's last
    /// token to prefill, so reuse ends at the block, and an entry stored
    /// under exactly the prefix is reused whole rather than trimmed by one,
    /// which a `MambaCache` cannot do. A hit from the cold tier is promoted to
    /// the hot one as for any request; that entry holds only text KV.
    func imageRequestLookupKey(_ tokens: [Int]) -> [Int]? {
        guard reusesPrefixBeforeMedia,
            let start = tokens.firstIndex(where: mediaTokenIDs.contains), start > 0
        else { return nil }
        return Array(tokens[...start])
    }

    /// Where an image request's remaining prefill splits: just past its last
    /// media token, when text follows. The part with the media goes through
    /// the model's `prepare` with the pixels; Gemma 4 runs that in one shot.
    /// The text after it then prefills on the warm cache in chunks, as a text
    /// request's does, so a long agent turn after an image no longer holds
    /// the whole prompt's attention at once. Attention there is causal, so
    /// splitting changes no position or mask; a block's bidirectional
    /// attention stays within the first part.
    ///
    /// Gemma 4 only: it positions from the cache offset alone. Qwen3.5's text
    /// after an image needs the rope delta the image leaves, which nothing
    /// carries across a split yet. Nil for another family, a text-only input,
    /// or one ending in its media.
    func mediaPrefillSplit(_ input: LMInput) -> Int? {
        guard family == .gemma4, Self.carriesMedia(input), input.text.tokens.ndim == 2 else { return nil }
        let ids = input.text.tokens[0].asArray(Int.self)
        guard let last = ids.lastIndex(where: mediaTokenIDs.contains), last + 1 < ids.count else {
            return nil
        }
        return last + 1
    }

    /// The generator this family's processor renders `.chat` input with when
    /// the configuration names none (`Gemma4Processor` and
    /// `Qwen3VLProcessor` build theirs directly). The factory wraps the
    /// processor in the configuration's generator when there is one, so the
    /// engine takes that first, as it does for an LLM.
    var processorMessageGenerator: any MessageGenerator {
        switch family {
        case .gemma4: Gemma4MessageGenerator()
        case .qwen35: Qwen3VLMessageGenerator()
        }
    }

    /// The generator a full-path load renders with when its configuration
    /// names none: ``processorMessageGenerator`` for messages with media, the
    /// text load's for the rest (``FullPathMessageGenerator``).
    var messageGenerator: any MessageGenerator {
        FullPathMessageGenerator(media: processorMessageGenerator)
    }

    /// Upstream's `qwen35.ropeDeltas` state key (private there, in
    /// `MLXVLM/Models/Qwen35.swift`), rebuilt from its id.
    static let qwen35RopeDeltasKeyID = "qwen35.ropeDeltas"

    /// The state every `TokenIterator` this load runs starts with, warm cache
    /// or cold. Qwen3.5 resumes a warm cache only with the rope delta its
    /// cached prefix accumulated: without one, `prepare` throws
    /// `missingState` and a decode step trips a `precondition`. Every prefix
    /// this path caches is image-free, and so is the prefix an image request
    /// reuses (``imageRequestLookupKey(_:)``), so the delta is zero; a cold
    /// prefill ignores it. Gemma 4 positions from the cache offset alone and needs
    /// none. Built per call, since `LMOutput.State` holds an `MLXArray`.
    func iteratorState() -> LMOutput.State? {
        switch family {
        case .gemma4:
            return nil
        case .qwen35:
            var state = LMOutput.State()
            state[LMOutput.Key<MLXArray>(Self.qwen35RopeDeltasKeyID)] = MLXArray(Int32(0))
            return state
        }
    }

    /// `input` as the text generation path expects a text-only prompt: `[N]`
    /// tokens and no mask. The VLM processors return `[1, N]` tokens and an
    /// all-ones mask even without an image, and the cache, the history
    /// boundary and the draft all slice axis 0. An input carrying an image,
    /// a video or audio is returned unchanged.
    static func textOnlyInput(_ input: LMInput) -> LMInput {
        guard !carriesMedia(input) else { return input }
        let tokens = input.text.tokens
        guard tokens.ndim == 2, tokens.dim(0) == 1 else {
            return input.text.mask == nil ? input : LMInput(tokens: tokens)
        }
        return LMInput(tokens: tokens.reshaped(-1))
    }

    /// Whether the processor attached pixels or audio to `input`.
    static func carriesMedia(_ input: LMInput) -> Bool {
        input.image != nil || input.video != nil || input.audio != nil
    }
}
