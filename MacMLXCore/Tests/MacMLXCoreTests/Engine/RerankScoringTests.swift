import MLXLMCommon
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
        #expect(rankAndTruncate(scores: [Float](), topN: 3).isEmpty)
    }

    // MARK: - HummingbirdServer.rerankResults (endpoint result shaping)

    @Test
    func rerankResultsAppliesScoreTransformAndPreservesRankOrder() {
        // A ranked (index, rawScore) list as rankAndTruncate would produce,
        // on the cosine path's Float scores with the Double(_) widening.
        let ranked: [(index: Int, score: Float)] = [(index: 2, score: 0.25), (index: 0, score: -1.0)]
        let results = HummingbirdServer.rerankResults(
            ranked: ranked, documents: ["a", "b", "c"],
            returnDocuments: false, scoreTransform: { Double($0) })
        #expect(results.map { $0.index } == [2, 0])
        #expect(abs(results[0].relevanceScore - 0.25) < 1e-9)
        #expect(results[1].relevanceScore == -1.0)
        // Documents omitted when not requested.
        #expect(results.allSatisfy { $0.document == nil })
    }

    /// Equal scores keep ascending index order, so a saturated sigmoid (two
    /// distinct logits mapping to the same `Double`) still ranks
    /// deterministically.
    @Test
    func rankAndTruncateBreaksTiesByAscendingIndex() {
        let scores: [Double] = [1.0, 0.5, 1.0, 0.5]
        #expect(rankAndTruncate(scores: scores).map { $0.index } == [0, 2, 1, 3])
    }

    /// `/v1/rerank` refuses a blank query or document before touching a
    /// model; whitespace-only counts as blank, as upstream's own check does.
    @Test
    func rerankInputProblemNamesTheBlankField() {
        #expect(HummingbirdServer.rerankInputProblem(query: "q", documents: ["a", "b"]) == nil)
        #expect(HummingbirdServer.rerankInputProblem(query: "", documents: ["a"]) == "query must not be empty")
        #expect(HummingbirdServer.rerankInputProblem(query: " \n", documents: ["a"]) == "query must not be empty")
        #expect(HummingbirdServer.rerankInputProblem(query: "q", documents: ["a", "  ", "c"]) == "documents[1] must not be empty")
        #expect(HummingbirdServer.rerankInputProblem(query: "q", documents: []) == nil)
    }

    /// Upstream's request-shaped errors become ``RerankRequestError`` (a 400
    /// at the endpoint); its model-side errors do not (they stay a 500).
    @Test
    func rerankRequestErrorMapsOnlyTheRequestShapedUpstreamCases() {
        #expect(RerankRequestError(RerankerError.emptyQuery) == .emptyQuery)
        #expect(RerankRequestError(RerankerError.emptyDocument(index: 2)) == .emptyDocument(index: 2))
        #expect(
            RerankRequestError(RerankerError.tooManyDocuments(actual: 65, maximum: 64))
                == .tooManyDocuments(actual: 65, maximum: 64))
        #expect(
            RerankRequestError(RerankerError.inputTooLong(actual: 9000, maximum: 8192))
                == .inputTooLong(actual: 9000, maximum: 8192))
        #expect(RerankRequestError(RerankerError.nonFiniteScore(index: 0, score: .nan)) == nil)
        #expect(RerankRequestError(RerankerError.invalidLogitShape([1, 2, 3])) == nil)
        #expect(
            RerankRequestError.tooManyDocuments(actual: 65, maximum: 64).errorDescription
                == "65 documents sent, but this reranker scores at most 64 per request")
    }

    /// The reranker path hands `Double` scores straight through: the same
    /// helpers rank them and expose them unchanged as `relevance_score`.
    @Test
    func rankAndTruncateAndRerankResultsWorkOnDoubleScores() {
        let scores: [Double] = [0.000013, 0.999856, 0.999931]
        let ranked = rankAndTruncate(scores: scores, topN: 2)
        #expect(ranked.map { $0.index } == [2, 1])
        let results = HummingbirdServer.rerankResults(
            ranked: ranked, documents: ["a", "b", "c"],
            returnDocuments: true, scoreTransform: { $0 })
        #expect(results.map { $0.relevanceScore } == [0.999931, 0.999856])
        #expect(results.map { $0.document } == ["c", "b"])
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
