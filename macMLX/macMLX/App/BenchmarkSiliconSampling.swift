// BenchmarkSiliconSampling.swift
// macMLX
//
// The slice of `SiliconMonitor` a benchmark run uses: reference-counted
// sampling control plus the two live readings its per-run collector folds.
//
// Same rationale as `BenchmarkEngineProviding` — an injection seam for
// `BenchmarkViewModel`, with `AppState` still passing the single shared
// `SiliconMonitor`. Pairing `activateSampling()` with `deactivateSampling()`
// on every exit path is the invariant this seam exists to make observable.

import Foundation
import MacMLXCore

@MainActor
protocol BenchmarkSiliconSampling: AnyObject, Sendable {

    /// Current bottleneck verdict, `nil` until the classifier has enough
    /// frames of the current generation to commit to one.
    var verdict: BottleneckVerdict? { get }

    /// Most recent hardware sample, `nil` before the first one lands.
    var latestSample: SiliconSample? { get }

    /// Register a consumer of the ~1 Hz sampling loop.
    func activateSampling()

    /// Release a consumer of the sampling loop. Must be called exactly once
    /// per `activateSampling()`.
    func deactivateSampling()
}

extension SiliconMonitor: BenchmarkSiliconSampling {}
