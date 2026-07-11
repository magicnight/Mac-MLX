import Foundation
import MLXLMCommon

// GLM-DSA (`model_type: glm_moe_dsa`, the GLM-5.1 family) — macMLX overlay,
// THEORETICAL TIER.
//
// ## Mapping
// `glm_moe_dsa.py` is a 54-line subclass: `class Model(DSV32Model)` that
// overrides nothing. It reuses our own DeepSeek V3.2 port (`DeepseekV32Model`)
// wholesale and differs ONLY in `ModelArgs` defaults — so there is no new model
// class. The overlay decodes a `GlmMoeDsaConfiguration`, which yields a
// `DeepseekV32Configuration` with the GLM-DSA defaults applied, then builds a
// stock `DeepseekV32Model`.
//
// ## Difference surface vs base DeepSeek V3.2 (field-by-field vs the Python
// `ModelArgs`)
//   1. `indexer_rope_interleave` defaults **true** (V3.2 defaults false). This
//      is the one default that reaches a numeric path: the lightning indexer's
//      RoPE runs interleaved (`traditional`). Pinned by
//      `GlmMoeDsaIndexerParityTests` against a captured `interleave=True`
//      fixture (see `docs/reference/capture_glm_moe_dsa_indexer.py`).
//   2. RoPE config is carried in a nested `rope_parameters` dict. GLM-DSA's
//      `__post_init__` copies it onto the base fields the V3.2 decoder reads:
//      `rope_theta = rope_parameters["rope_theta"]` and
//      `rope_scaling = rope_parameters`. Base V3.2 reads top-level
//      `rope_theta` / `rope_scaling`, so without this the RoPE base would fall
//      back to the wrong default.
// Every other GLM-DSA `ModelArgs` field matches a `DeepseekV32Configuration`
// field 1:1 (verified against the real `glm_moe_dsa` config.json).
//
// ## GLM-5.2 exclusion
// GLM-5.2 adds "IndexShare": a per-layer `full`/`shared` indexer schedule
// (`indexer_types`, with `index_topk_freq` / `index_skip_topk_offset`), still
// pending upstream (mlx-lm #1410). A config carrying those fields would silently
// load onto the plain single-indexer path with the wrong topology, so it is
// rejected loudly instead.
//
// ## Untested on real weights
// THEORETICAL TIER — architecture verified via fixture parity (config + the
// interleaved-RoPE indexer difference component); never smoke-tested on real
// weights (smallest quant exceeds the size budget); model-specific issues are
// handled issue-driven.

/// Decodes a `glm_moe_dsa` config.json into the equivalent
/// `DeepseekV32Configuration`, applying GLM-DSA's `ModelArgs` defaults and
/// rejecting GLM-5.2 IndexShare configs.
struct GlmMoeDsaConfiguration: Decodable {

    /// The DeepSeek V3.2 configuration with GLM-DSA defaults applied — this is
    /// what `DeepseekV32Model` is built from.
    let base: DeepseekV32Configuration

    /// GLM-DSA-specific keys read on top of the base V3.2 schema.
    private enum GlmKey: String, CodingKey {
        case indexerTypes = "indexer_types"
        case indexTopkFreq = "index_topk_freq"
        case indexSkipTopkOffset = "index_skip_topk_offset"
        case indexerRopeInterleave = "indexer_rope_interleave"
        case ropeParameters = "rope_parameters"
    }

    init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: GlmKey.self)

        // GLM-5.2 IndexShare guard — the plain path has no per-layer indexer
        // schedule, so refuse rather than mis-load.
        //
        // Field names GROUND-TRUTHED (2026-07-11) against BOTH sources:
        // mlx-lm PR #1410's ModelArgs additions (`indexer_types`,
        // `index_topk_pattern`, `index_topk_freq`, `index_skip_topk_offset`)
        // and the real zai-org/GLM-5.2-FP8 config.json, which carries all
        // three checked keys with concrete values (`indexer_types` array,
        // `index_topk_freq: 4`, `index_skip_topk_offset: 3`) — any one hit
        // rejects, so a real 5.2 config can never slip through. GLM-5.1
        // configs carry none of these keys (verified) and pass untouched.
        if container.contains(.indexerTypes) || container.contains(.indexTopkFreq)
            || container.contains(.indexSkipTopkOffset)
        {
            throw ModelOverlayError.glmDsaIndexShareUnsupported
        }

        // Base fields via DeepSeek V3.2's lenient decoder.
        var base = try DeepseekV32Configuration(from: decoder)

        // (1) Real GLM-DSA configs SHIP `indexer_rope_interleave: true`
        // (verified against zai-org checkpoints; the Python ModelArgs does not
        // declare the field at all). Honor an explicit value, and default to
        // true DEFENSIVELY when absent — the opposite of V3.2's false.
        base.indexerRopeInterleave =
            try container.decodeIfPresent(Bool.self, forKey: .indexerRopeInterleave) ?? true

        // (2) __post_init__: derive rope_theta / rope_scaling from the nested
        // rope_parameters dict when present.
        if let ropeParameters = try container.decodeIfPresent(
            [String: StringOrNumber].self, forKey: .ropeParameters)
        {
            if let theta = ropeParameters["rope_theta"]?.asFloat() {
                base.ropeTheta = theta
            }
            base.ropeScaling = ropeParameters
        }

        self.base = base
    }
}
