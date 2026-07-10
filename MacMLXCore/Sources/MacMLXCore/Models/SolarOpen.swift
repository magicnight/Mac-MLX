import Foundation

// Solar-Open-100B (`model_type: solar_open`) — macMLX overlay, THEORETICAL TIER.
//
// ## Mapping
// `solar_open.py` is a 38-line re-skin: `from .glm4_moe import Model`. The
// runtime architecture is upstream `mlx-swift-lm`'s `GLM4MoEModel` used
// unchanged (its `init` is `public`, so the overlay constructs it directly).
// solar_open contributes ONLY a `ModelArgs` dataclass, and its sole effect is
// to *default* a handful of fields that `GLM4MoEConfiguration.init(from:)`
// decodes as required. Solar-Open's shipped `config.json` omits exactly those
// fields, so `patchedConfigData` injects the Python defaults before the stock
// `GLM4MoEConfiguration` decode.
//
// ## Difference surface (the only thing that can go wrong)
// Config parsing. There is zero numerical difference from GLM4MoE — the model
// is reused byte-for-byte — so forward parity is already owned by upstream
// GLM4MoE; Solar's fixture is the config-injection round-trip
// (`SolarOpenConfigurationTests`).
//
// ## Untested on real weights
// THEORETICAL TIER — architecture verified via fixture parity; never
// smoke-tested on real weights (smallest quant exceeds the size budget);
// model-specific issues are handled issue-driven. Badge wiring on
// LocalModel/ModelCard is a separate follow-up; this is the code-layer gate.

/// Config-adaptation namespace for `solar_open`. No model type of its own — the
/// overlay decodes a `GLM4MoEConfiguration` from the patched data and builds a
/// stock `GLM4MoEModel`.
enum SolarOpen {

    /// Return `data` with `solar_open.ModelArgs`'s defaults filled in for any
    /// missing key, so the result decodes as a stock `GLM4MoEConfiguration`.
    ///
    /// Only the fields that `solar_open.ModelArgs` defaults **and**
    /// `GLM4MoEConfiguration.init(from:)` requires are injected; a key already
    /// present in the config always wins (an explicit value is never
    /// overwritten). Real Hugging Face `config.json` files are strict JSON, so
    /// `JSONSerialization` parses them; a non-object payload is rejected loudly.
    static func patchedConfigData(_ data: Data) throws -> Data {
        guard let object = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw ModelOverlayError.solarOpenMalformedConfig
        }

        // Mirrors `solar_open.ModelArgs` field defaults (kept local so the
        // heterogeneous `[String: Any]` never becomes a non-Sendable global).
        let injectedDefaults: [String: Any] = [
            "attention_bias": false,
            "use_qk_norm": false,
            "n_group": 1,
            "topk_group": 1,
            "scoring_func": "sigmoid",
            "topk_method": "noaux_tc",
        ]

        var patched = object
        for (key, value) in injectedDefaults where patched[key] == nil {
            patched[key] = value
        }
        return try JSONSerialization.data(withJSONObject: patched)
    }
}
