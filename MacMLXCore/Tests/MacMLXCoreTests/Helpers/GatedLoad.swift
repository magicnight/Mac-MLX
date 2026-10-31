import Foundation

@testable import MacMLXCore

/// Which load the gated real-weights suites put their Gemma 4 / Qwen3.6
/// checkpoint on. By default the text load (`.mlx`, `LLMModelFactory`), as
/// they always have. With `MACMLX_GATED_FULL_PATH_VLM=1` (passed to
/// xcodebuild's runner as `TEST_RUNNER_MACMLX_GATED_FULL_PATH_VLM=1`) the
/// same checkpoint loads through the VLM factory and is served on the text
/// generation path through the engine's forcing seam, before its family is
/// in ``FullPathVLM/servedFamilies``. A family flips only once these suites
/// pass on that load too (the parity gate in
/// `docs/superpowers/specs/2026-10-31-gemma4-qwen35-vision-design.md`).
enum GatedLoad {

    static var isFullPathVLM: Bool {
        ProcessInfo.processInfo.environment["MACMLX_GATED_FULL_PATH_VLM"] == "1"
    }

    /// The format a suite's checkpoint is loaded as.
    static var format: ModelFormat { isFullPathVLM ? .mlxVLM : .mlx }

    /// Arm `engine` for this load. Call before `engine.load`.
    static func prepare(_ engine: MLXSwiftEngine) async {
        await engine.setForcesFullPathVLM(isFullPathVLM)
    }

    /// Load `id` from `directory` on `engine` as this run's load, and check
    /// the forced load took the full path, so a run meant to test it cannot
    /// pass on the stock VLM path.
    static func load(_ engine: MLXSwiftEngine, id: String, directory: URL) async throws {
        await prepare(engine)
        try await engine.load(LocalModel(
            id: id, displayName: id, directory: directory, sizeBytes: 0,
            format: format, quantization: nil, parameterCount: nil, architecture: nil))
        try await requireFullPathIfForced(engine)
    }

    /// Throws when this run forces the full path and `engine`'s resident
    /// model is not on it.
    static func requireFullPathIfForced(_ engine: MLXSwiftEngine) async throws {
        guard isFullPathVLM else { return }
        guard await engine.servesFullPathVLM else {
            throw CocoaError(
                .featureUnsupported,
                userInfo: [NSLocalizedDescriptionKey: "MACMLX_GATED_FULL_PATH_VLM=1 but the load is not on the full path"])
        }
    }
}
