// BenchmarkEngineProviding.swift
// macMLX
//
// The slice of `EngineCoordinator` a benchmark run actually uses.
//
// Extracted as a protocol purely to give `BenchmarkViewModel` an injection
// seam: `AppState` still hands it the one real `EngineCoordinator`, and every
// member below forwards to the identical coordinator API it called before, so
// the production path is unchanged — same object, same calls, same order.
//
// `Sendable` is inherited because `BenchmarkViewModel.doRun` captures the
// coordinator into the `@Sendable` generate closure it hands the runner (the
// concrete `EngineCoordinator` is implicitly `Sendable` as a `@MainActor`
// final class; an existential needs to say so).

import Foundation
import MacMLXCore

@MainActor
protocol BenchmarkEngineProviding: AnyObject, Sendable {

    /// Model the coordinator currently treats as active, `nil` when none is.
    var currentModel: LocalModel? { get }

    /// Engine lifecycle state. The benchmark only reads `isLoaded`, to decide
    /// whether it has to pay a cold load before timing anything.
    var status: EngineStatus { get }

    /// Provenance stamped onto the saved `BenchmarkResult`.
    var engineID: EngineID { get }
    var engineVersion: String { get }

    /// Load `model`, returning the coordinator's own result. The benchmark
    /// ignores the failure case deliberately: a failed load surfaces as the
    /// generation error that follows, with the engine's own message.
    @discardableResult
    func load(_ model: LocalModel, adapter: LocalAdapter?) async -> Result<Void, Error>

    /// Stream tokens for `request` against the active engine.
    func generate(_ request: GenerateRequest) -> AsyncThrowingStream<GenerateChunk, Error>
}

extension EngineCoordinator: BenchmarkEngineProviding {}
