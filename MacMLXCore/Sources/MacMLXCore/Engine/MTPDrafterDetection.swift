import Foundation
import MLXLMCommon

/// Cheap, load-free classification of a draft model as an **MTP (Multi-Token
/// Prediction) block drafter** rather than a classic draft model.
///
/// ## Status: groundwork, not yet consumed by the engine
///
/// The intended shape is to upgrade macMLX's single speculative-decoding
/// surface (`draft_model` / `GenerateRequest.draftModelID`) automatically:
/// when the named drafter is an MTP drafter, route generation to
/// `MTPSpeculativeTokenIterator` (shared K/V with the target, one block of
/// candidates per round) instead of the classic `SpeculativeTokenIterator` —
/// no new request field, no new GUI switch. That routing decision has to be
/// made BEFORE a factory is chosen, i.e. without loading multi-GB weights,
/// which is what this type answers.
///
/// The engine wiring is deliberately NOT landed yet, because MTP cannot fire
/// for any checkpoint macMLX can currently load:
///
/// - `MTPSpeculativeTokenIterator` needs the TARGET model to emit
///   `mtpLastHiddenStatesKey` + `mtpSharedKVStatesKey` in its `LMOutput.State`;
///   without them it latches into single-token passthrough on the first round.
/// - In mlx-swift-lm the only model that emits them is `Gemma4` in **MLXVLM**
///   (via its `callAsFunction(_:cache:state:)` overload). MLXLLM's
///   `Gemma4Model` has no emit path at all.
/// - macMLX classifies a Gemma 4 checkpoint as `.mlx` (`gemma4` is not in
///   `ModelLibraryManager.knownVLMTypes`), so it loads through
///   `LLMModelFactory` as the non-emitting `Gemma4Model`.
///
/// Wiring the iterator up before that is fixed would cost a resident drafter
/// and a bypassed prompt cache while never once speculating — strictly slower
/// than leaving it off. Reaching it needs the target to load through the VLM
/// factory AND text-only requests to stop short-circuiting past draft-model
/// resolution, or an upstream change that moves the emit hook into MLXLLM.
///
/// ## The signal
///
/// `config.json`'s top-level `model_type`, looked up in
/// `MTPDrafterTypeRegistry.shared` — the exact registry (and the exact key)
/// `MTPDrafterModelFactory._load` consults when it instantiates the drafter,
/// so a `true` here means the factory will find a creator. The registry is
/// populated by `ModelOverlay.registerAll()`; before that runs it is empty and
/// everything classifies as a classic draft model.
///
/// `MTPDrafterRegistry` (upstream's list of KNOWN drafter ids, e.g.
/// `mlx-community/gemma-4-31B-it-assistant-bf16`) is deliberately NOT
/// consulted: those are hub-style ids containing `/`, which
/// `MLXSwiftEngine.isValidDraftModelID` rejects outright, so the check could
/// never fire on this path.
///
/// ## Failure direction
///
/// Any read/parse failure (missing `config.json`, unreadable bytes, JSON the
/// stock `JSONSerialization` parser rejects) classifies as "not an MTP
/// drafter". Under the classic path that is where a genuine drafter already
/// ends up today, and `LLMModelFactory.loadContainer` throws
/// `unsupportedModelType` for it — loud, never silently wrong tokens.
enum MTPDrafterDetection {

    /// The `model_type` declared by the `config.json` in `directory`, or nil
    /// when the file is absent/unreadable or carries no top-level
    /// `model_type`.
    ///
    /// Reads the top-level key only — matching `BaseConfiguration.modelType`,
    /// which is what `MTPDrafterModelFactory._load` dispatches on. Mirrors
    /// `MLXSwiftEngine.inferToolCallFormatFallback`'s `JSONSerialization`
    /// approach (any IO/JSON error is treated as "no answer").
    static func modelType(inModelDirectory directory: URL) -> String? {
        let configURL = directory.appending(component: "config.json")
        guard let data = try? Data(contentsOf: configURL),
              let root = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else {
            return nil
        }
        return root["model_type"] as? String
    }

    /// Whether the model checkpoint at `directory` is an MTP drafter — i.e.
    /// its `model_type` has a creator registered in
    /// `MTPDrafterTypeRegistry.shared`.
    static func isMTPDrafter(directory: URL) async -> Bool {
        guard let type = modelType(inModelDirectory: directory) else { return false }
        return await MTPDrafterTypeRegistry.shared.contains(type)
    }

    /// Whether the draft model named by `modelID` is an MTP drafter.
    ///
    /// Resolves the id through `MLXSwiftEngine.draftModelDirectory(id:)` — the
    /// same path-traversal-guarded resolution the classic draft loader uses —
    /// so a rejected (unsafe) id simply classifies as "not an MTP drafter"
    /// and the classic loader raises the id error itself, keeping exactly one
    /// place responsible for reporting it.
    static func isMTPDrafter(modelID: String) async -> Bool {
        guard let directory = try? MLXSwiftEngine.draftModelDirectory(id: modelID) else {
            return false
        }
        return await isMTPDrafter(directory: directory)
    }
}
