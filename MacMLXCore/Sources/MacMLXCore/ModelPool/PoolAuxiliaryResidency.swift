import Foundation

/// `AuxiliaryResidency` over a `ModelPool` (#130): what the app hands the
/// server. Admission reads the model's TTL from `ModelParametersStore`, the
/// per-model setting chat models use, and everything else goes straight to
/// the pool, off the main actor.
public struct PoolAuxiliaryResidency: AuxiliaryResidency {
    public let pool: ModelPool
    /// Where the model's TTL comes from; the app's store by default.
    public let parameters: ModelParametersStore

    public init(pool: ModelPool, parameters: ModelParametersStore = ModelParametersStore()) {
        self.pool = pool
        self.parameters = parameters
    }

    public func admit(
        _ kind: AuxiliaryKind,
        _ model: LocalModel,
        load: @Sendable () async throws -> Void,
        release: @escaping @Sendable () async -> Void
    ) async throws {
        let parameters = await parameters.load(for: model.id)
        try await pool.admitAuxiliary(
            kind, model, ttlSeconds: parameters.ttlSeconds, inUse: true, load: load, release: release)
    }

    public func beginUse(_ kind: AuxiliaryKind, _ modelID: String) async -> Bool {
        await pool.beginUsingAuxiliary(kind, modelID)
    }

    public func endUse(_ kind: AuxiliaryKind, _ modelID: String) async {
        await pool.endUsingAuxiliary(kind, modelID)
    }

    public func forget(_ kind: AuxiliaryKind, _ modelID: String) async {
        await pool.forgetAuxiliary(kind, modelID)
    }
}
