import Foundation

/// How the server reports its reranker and embedder to the model pool one
/// layer up (#130). The server owns those engines and swaps them under its
/// generation lock; the pool budgets their bytes, evicts them under pressure
/// and sweeps them when idle like any chat model, and makes the server let go
/// through the `release` it was handed at admission. Absent (the CLI, which
/// has no pool), the server loads them directly and nothing accounts for them.
///
/// A request that admits a model, or finds it resident, is about to compute
/// on it: `admit` returns with the model in use, `beginUse` marks a resident
/// one in use in a single step with the pool and says whether the pool still
/// held it, and `endUse` ends the use. A false from `beginUse` means the pool
/// has let the model go (an eviction or sweep, whether or not its release has
/// reached the server); the server then swaps rather than computes on an
/// engine the pool no longer counts.
///
/// `PoolAuxiliaryResidency` is the implementation over a `ModelPool`.
public protocol AuxiliaryResidency: Sendable {
    /// Make room for `model` and reserve its cost, then run `load`; on success
    /// keep `release` as the way to reclaim the model, in use until `endUse`.
    /// Throws what `load` throws, with nothing reserved. `release` must
    /// capture nothing that keeps the engine alive: the pool holds it for as
    /// long as the entry lives, and a release that pins the engine would
    /// defeat its purpose. It must not take the server's generation lock
    /// either: the pool runs a release while a handler holding the lock
    /// awaits `admit`.
    func admit(
        _ kind: AuxiliaryKind,
        _ model: LocalModel,
        load: @Sendable () async throws -> Void,
        release: @escaping @Sendable () async -> Void
    ) async throws

    /// A request found the model resident and is about to compute on it:
    /// refresh its idle clock and mark it in use, in one step, so the pool
    /// spares it from eviction and the idle sweep (POOL-3). False when the
    /// pool no longer holds it; nothing is marked then.
    func beginUse(_ kind: AuxiliaryKind, _ modelID: String) async -> Bool

    /// The request has finished with the model.
    func endUse(_ kind: AuxiliaryKind, _ modelID: String) async

    /// The server let go of the model on its own (a swap, its unload door, a
    /// stop, an admission it did not install): drop the entry without
    /// running its release.
    func forget(_ kind: AuxiliaryKind, _ modelID: String) async
}
