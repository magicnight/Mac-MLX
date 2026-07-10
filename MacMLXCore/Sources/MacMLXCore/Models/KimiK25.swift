import Foundation
import MLXLLM

// Kimi K2 / K2.5 (`model_type: kimi_k25`) — macMLX overlay, THEORETICAL TIER,
// currently BLOCKED on an upstream access level.
//
// ## Mapping
// `kimi_k25.py` is an 83-line VLM shell over DeepSeek V3. The text tower is
// upstream `DeepseekV3Model` living under a `language_model` module; the config
// nests the full DeepSeek V3 args under `text_config`. Its only real work is a
// `sanitize` that strips the vision weights (`vision_tower` / `vision_model` /
// `multi_modal_projector` / `mm_projector`) and then defers to DeepSeek V3's
// own sanitize on the `language_model.*` subtree.
//   • K2 (original, non-vision) is plain `deepseek_v3` — the community loads it
//     by renaming the model_type, and the stock factory already handles it. So
//     there is nothing for the overlay to add for K2 itself.
//   • K2.5 has its own `model_type: kimi_k25` (vision-augmented checkpoint) and
//     is what this file targets.
//
// ## Blocked (why there is no `KimiK25Model` here)
// The wrapper would nest a `DeepseekV3Model` under the `language_model` key so
// the checkpoint's `language_model.model.*` / `language_model.lm_head.*` layout
// maps 1:1, forwarding `callAsFunction` / `kvHeads` / `loraLayers` and composing
// sanitize (strip vision → strip the `language_model.` prefix →
// `DeepseekV3Model.sanitize` → re-add the prefix). That is a ~40-line
// composition — but upstream `mlx-swift-lm` declares `DeepseekV3Model.init(_:)`
// and `DeepseekV3ModelInner.init` as `internal`, so this overlay (a separate
// module) cannot construct the text core. The registry creator is synchronous,
// so the async `LLMTypeRegistry.createModel` escape hatch is unavailable too.
// Solar-Open works precisely because `GLM4MoEModel.init` is `public`.
//
// The `kimi_k25` creator therefore decodes `KimiK25Configuration` (proving the
// config path is correct and ready) and then throws
// `ModelOverlayError.kimiK25RequiresPublicDeepseekV3Init` — a loud, actionable
// failure, never a silent or subtly-wrong load. It unblocks the instant
// upstream marks the initializer `public` (a one-line change / upstream PR),
// after which the wrapper drops in unchanged.
//
// THEORETICAL TIER — config half verified via fixture parity; the model is not
// buildable on the pinned upstream and is never smoke-tested on real weights;
// model-specific issues are handled issue-driven.

/// Decodes a `kimi_k25` config.json: the text tower is the DeepSeek V3 args
/// nested under `text_config`. This is the finished half of the Kimi port; the
/// wrapping model is blocked on upstream (see file header).
struct KimiK25Configuration: Decodable {

    /// The DeepSeek V3 text-tower configuration, nested under `text_config`.
    let textConfig: DeepseekV3Configuration

    enum CodingKeys: String, CodingKey {
        case textConfig = "text_config"
    }
}
