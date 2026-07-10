import Foundation

/// Errors raised by `ModelOverlay` creators when a `model_type` is recognized
/// but cannot (yet) be instantiated by the overlay.
///
/// These are deliberate, *loud* failures — the "theoretical tier" contract is
/// that the code layer never silently mis-loads a model. When an architecture
/// is registered but a specific variant or upstream dependency is missing, the
/// creator throws one of these with an actionable message instead of returning
/// a subtly wrong model.
public enum ModelOverlayError: Error, LocalizedError, Equatable {

    /// GLM-5.2's "IndexShare" DSA schedule (per-layer `full`/`shared` indexer
    /// sharing) is not yet ported. The upstream mlx-lm implementation is
    /// pending in PR #1410; until it lands, a config carrying `indexer_types`
    /// (or its `index_topk_freq` / `index_skip_topk_offset` companions) would
    /// load onto the plain `glm_moe_dsa` path with the wrong indexer topology.
    case glmDsaIndexShareUnsupported

    /// Kimi K2.5's text core is upstream `mlx-swift-lm`'s `DeepseekV3Model`,
    /// whose initializer is `internal` (unlike `GLM4MoEModel`, which is
    /// `public`). A separate module — this overlay — therefore cannot construct
    /// it to build the `language_model` wrapper. The port is otherwise complete
    /// (`KimiK25Configuration` decodes the nested `text_config`); it unblocks
    /// the moment upstream marks `DeepseekV3Model.init(_:)` `public`.
    case kimiK25RequiresPublicDeepseekV3Init

    /// A `solar_open` config.json could not be parsed as a JSON object while
    /// injecting the `solar_open` ModelArgs defaults.
    case solarOpenMalformedConfig

    public var errorDescription: String? {
        switch self {
        case .glmDsaIndexShareUnsupported:
            return "GLM-5.2 IndexShare not yet supported (upstream mlx-lm #1410 pending)"
        case .kimiK25RequiresPublicDeepseekV3Init:
            return
                "Kimi K2.5 blocked: upstream mlx-swift-lm DeepseekV3Model.init is internal (needs public) so the overlay cannot construct the DeepSeek V3 text core"
        case .solarOpenMalformedConfig:
            return "Solar-Open config.json is not a JSON object"
        }
    }
}
