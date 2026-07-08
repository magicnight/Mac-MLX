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
}
