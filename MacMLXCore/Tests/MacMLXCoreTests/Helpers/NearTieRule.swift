// Copyright © 2026 macMLX. English comments only.

import XCTest

@testable import MacMLXCore

/// The rule two greedy runs of one prompt are compared under when their
/// prefills were split at different places, so they do not round alike (the
/// reasoning and the measurements are on `HistoryBoundaryPromptCacheSmokeTests`).
/// Up to the first token the runs disagree on, every chosen token's logprob
/// must agree within `logprobTolerance` and every top-5 alternative both runs
/// list, outside the far tail, within `alternativeTolerance`; at that token
/// each run's choice must be within `nearTieTolerance` of the other run's
/// choice in its own distribution; after it the continuations differ and are
/// not compared.
struct NearTieRule {

    /// Bound on the logprob difference between two runs before they first
    /// disagree: two bf16 steps at the logits' magnitude (measured: one).
    var logprobTolerance: Float = 0.25

    /// Bound on the difference of a top-5 alternative both runs list, before
    /// they first disagree. Alternatives move more than the chosen token
    /// (measured: up to 1.375 unmutated; with a position doubled, 3.375 among
    /// alternatives above `alternativeFloor`).
    var alternativeTolerance: Float = 2.0

    /// Alternatives both runs put below this logprob (e^-8, about 3e-4) are
    /// not compared: that far down the tail bf16 rounding alone moves them
    /// by whole units (Mellum2 measured 2.625 at -16 on a token whose choice
    /// had probability 1).
    var alternativeFloor: Float = -8

    /// Bound on the gap, in one run's distribution, between its own greedy
    /// token and the other run's at the first token they disagree on: four
    /// bf16 steps (measured: three on Qwen3.6).
    var nearTieTolerance: Float = 0.5

    /// One run: its text and the logprobs it captured (`topLogprobs` 5).
    struct Run {
        let name: String
        let text: String
        let logprobs: [TokenLogprob]
    }

    func assertMatches(
        _ first: Run, _ second: Run, _ label: String, file: StaticString = #filePath, line: UInt = #line
    ) {
        XCTAssertFalse(first.logprobs.isEmpty, "\(label): logprobs were captured", file: file, line: line)
        let pairs = Array(zip(first.logprobs, second.logprobs))
        let divergence = pairs.firstIndex { $0.0.token != $0.1.token }
        for (position, (a, b)) in pairs.prefix(divergence ?? pairs.count).enumerated() {
            XCTAssertLessThanOrEqual(
                abs(a.logprob - b.logprob), logprobTolerance,
                "\(label) token \(position) \(a.token.debugDescription): \(a.logprob) against \(b.logprob)",
                file: file, line: line)
            for alternative in a.topLogprobs {
                guard let other = b.topLogprobs.first(where: { $0.token == alternative.token }),
                    max(alternative.logprob, other.logprob) >= alternativeFloor
                else { continue }
                XCTAssertLessThanOrEqual(
                    abs(alternative.logprob - other.logprob), alternativeTolerance,
                    "\(label) token \(position), alternative \(alternative.token.debugDescription): "
                        + "\(alternative.logprob) against \(other.logprob)",
                    file: file, line: line)
            }
        }
        guard let divergence else {
            XCTAssertEqual(first.text, second.text, "\(label): the same output", file: file, line: line)
            return
        }
        let ends = [
            (first.name, first.logprobs[divergence], second.logprobs[divergence]),
            (second.name, second.logprobs[divergence], first.logprobs[divergence]),
        ]
        for (run, own, other) in ends {
            guard let alternative = own.topLogprobs.first(where: { $0.token == other.token }) else {
                XCTFail(
                    "\(label) token \(divergence): the \(run) run does not rank \(other.token.debugDescription) "
                        + "among its alternatives \(own.topLogprobs.map(\.token))",
                    file: file, line: line)
                continue
            }
            XCTAssertLessThanOrEqual(
                own.logprob - alternative.logprob, nearTieTolerance,
                "\(label) token \(divergence): the \(run) run picked \(own.token.debugDescription) at "
                    + "\(own.logprob), \(other.token.debugDescription) at \(alternative.logprob)",
                file: file, line: line)
        }
    }
}
