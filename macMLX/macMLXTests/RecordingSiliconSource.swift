// RecordingSiliconSource.swift
// macMLXTests
//
// Test double for `SiliconMonitor`'s benchmark seam. It carries no hardware and
// publishes no verdict — its whole job is to count the reference-counted
// sampling calls so a test can assert that a run balanced them.
//
// The real `SiliconMonitor` hides its `SamplingActivation` behind `private`, so
// an unbalanced pair is invisible from outside: an extra `activate` just leaves
// the ~1 Hz IOReport loop running forever, and an extra `deactivate` silently
// cuts sampling out from under whoever else still needs it. Counting here makes
// both observable.

import Foundation
import MacMLXCore
@testable import macMLX

final class RecordingSiliconSource: BenchmarkSiliconSampling {

    /// No verdict and no sample: the collector's admission rule then folds
    /// nothing, which is exactly the "run too short to attribute" case the view
    /// model must survive without claiming a bottleneck.
    var verdict: BottleneckVerdict? { nil }
    var latestSample: SiliconSample? { nil }

    private(set) var activateCount = 0
    private(set) var deactivateCount = 0

    /// Outstanding activations. Zero means every `activateSampling()` was paid
    /// back; negative means someone over-released.
    var balance: Int { activateCount - deactivateCount }

    func activateSampling() { activateCount += 1 }
    func deactivateSampling() { deactivateCount += 1 }
}
