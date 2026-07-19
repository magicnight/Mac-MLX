import Testing
import Foundation
@testable import MacMLXCore

/// Pure unit tests for the bi-encoder rerank scoring helpers. These use
/// fixed vectors so they need no real embedding model.
@Suite("Rerank scoring")
struct RerankScoringTests {

    @Test
    func cosineOfIdenticalUnitVectorsIsOne() {
        #expect(abs(cosineSimilarity([1, 0, 0], [1, 0, 0]) - 1) < 1e-6)
    }

    @Test
    func cosineOfOrthogonalVectorsIsZero() {
        #expect(abs(cosineSimilarity([1, 0], [0, 1])) < 1e-6)
    }

    @Test
    func cosineNormalizesNonUnitVectors() {
        // Same direction, different magnitudes → cosine still 1.
        #expect(abs(cosineSimilarity([2, 0], [5, 0]) - 1) < 1e-6)
    }

    @Test
    func cosineHandlesEmptyAndZeroVectors() {
        #expect(cosineSimilarity([], []) == 0)
        #expect(cosineSimilarity([0, 0], [1, 1]) == 0)
    }

    @Test
    func rerankOrdersByDescendingSimilarity() {
        let query: [Float] = [1, 0]
        let documents: [[Float]] = [
            [0, 1],       // index 0 — orthogonal, score 0
            [1, 0],       // index 1 — identical, score 1
            [1, 1],       // index 2 — 45°, score ~0.707
        ]
        let ranked = rerankByCosine(query: query, documents: documents)
        #expect(ranked.map { $0.index } == [1, 2, 0])
        // Scores are monotonically non-increasing.
        #expect(ranked[0].score >= ranked[1].score)
        #expect(ranked[1].score >= ranked[2].score)
    }

    @Test
    func rerankTruncatesToTopN() {
        let query: [Float] = [1, 0]
        let documents: [[Float]] = [
            [0, 1],       // index 0
            [1, 0],       // index 1
            [1, 1],       // index 2
        ]
        let ranked = rerankByCosine(query: query, documents: documents, topN: 2)
        #expect(ranked.count == 2)
        #expect(ranked.map { $0.index } == [1, 2])
    }

    @Test
    func rerankHandlesEmptyDocuments() {
        let ranked = rerankByCosine(query: [1, 0], documents: [], topN: 5)
        #expect(ranked.isEmpty)
    }

    @Test
    func rerankIgnoresOutOfRangeTopN() {
        let query: [Float] = [1, 0]
        let documents: [[Float]] = [[1, 0], [0, 1]]
        // topN larger than the document count returns everything.
        let ranked = rerankByCosine(query: query, documents: documents, topN: 99)
        #expect(ranked.count == 2)
    }

    // MARK: - rankAndTruncate (shared by cross-encoder + cosine paths)

    @Test
    func rankAndTruncateOrdersByDescendingScoreKeepingIndices() {
        // Raw cross-encoder logits (can be negative — unlike cosine).
        let scores: [Float] = [-2.0, 5.0, 0.5, 5.0]
        let ranked = rankAndTruncate(scores: scores)
        // Descending by score; original indices preserved. Ties (index 1 & 3,
        // both 5.0) may order arbitrarily, so assert the score sequence and the
        // tie SET rather than an exact tied order.
        #expect(ranked.map { $0.score } == [5.0, 5.0, 0.5, -2.0])
        #expect(Set([ranked[0].index, ranked[1].index]) == Set([1, 3]))
        #expect(ranked[2].index == 2)
        #expect(ranked[3].index == 0)
    }

    @Test
    func rankAndTruncateHonorsTopNAndIgnoresOutOfRange() {
        let scores: [Float] = [0.1, 0.9, 0.5]
        #expect(rankAndTruncate(scores: scores, topN: 2).map { $0.index } == [1, 2])
        // Out-of-range / negative topN returns the full ranking.
        #expect(rankAndTruncate(scores: scores, topN: 99).count == 3)
        #expect(rankAndTruncate(scores: scores, topN: -1).count == 3)
        #expect(rankAndTruncate(scores: [], topN: 3).isEmpty)
    }

    // MARK: - rerankSigmoid

    @Test
    func sigmoidIsHalfAtZeroMonotonicAndBounded() {
        #expect(abs(rerankSigmoid(0) - 0.5) < 1e-9)
        // Strictly monotonic — never reorders the raw-logit ranking.
        #expect(rerankSigmoid(-1) < rerankSigmoid(0))
        #expect(rerankSigmoid(0) < rerankSigmoid(1))
        // Saturating but bounded to [0, 1]. (At large magnitudes Double rounds
        // the tails to exactly 1.0 / 0.0, so assert `<=` / `>=` at ±50 and use
        // ±10 for the strictly-open interior bound.)
        #expect(rerankSigmoid(50) <= 1.0 && rerankSigmoid(10) > 0.99)
        #expect(rerankSigmoid(-50) >= 0.0 && rerankSigmoid(-10) < 0.01)
        #expect(rerankSigmoid(10) < 1.0 && rerankSigmoid(-10) > 0.0)
    }

    // MARK: - HummingbirdServer.rerankResults (endpoint result shaping)

    @Test
    func rerankResultsAppliesScoreTransformAndPreservesRankOrder() {
        // A ranked (index, rawScore) list as rankAndTruncate would produce.
        let ranked: [(index: Int, score: Float)] = [(index: 2, score: 0.0), (index: 0, score: -1.0)]
        let results = HummingbirdServer.rerankResults(
            ranked: ranked, documents: ["a", "b", "c"],
            returnDocuments: false, scoreTransform: rerankSigmoid)
        #expect(results.map { $0.index } == [2, 0])
        #expect(abs(results[0].relevanceScore - 0.5) < 1e-9)  // sigmoid(0)
        #expect(results[1].relevanceScore < 0.5)              // sigmoid(-1)
        // Documents omitted when not requested.
        #expect(results.allSatisfy { $0.document == nil })
    }

    @Test
    func rerankResultsEchoesDocumentsByOriginalIndexWhenRequested() {
        let ranked: [(index: Int, score: Float)] = [(index: 2, score: 0.9), (index: 0, score: 0.1)]
        let results = HummingbirdServer.rerankResults(
            ranked: ranked, documents: ["zero", "one", "two"],
            returnDocuments: true, scoreTransform: { Double($0) })
        // Echoed document tracks the ORIGINAL index, not the rank position.
        #expect(results[0].document == "two")
        #expect(results[1].document == "zero")
    }

    @Test
    func rerankResultsSkipsOutOfRangeDocumentIndex() {
        let ranked: [(index: Int, score: Float)] = [(index: 5, score: 0.9)]
        let results = HummingbirdServer.rerankResults(
            ranked: ranked, documents: ["only"],
            returnDocuments: true, scoreTransform: { Double($0) })
        // Index 5 has no document → nil rather than a crash.
        #expect(results[0].document == nil)
    }
}
