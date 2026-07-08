import Foundation

// MARK: - Bi-encoder rerank scoring

/// Cosine similarity between two vectors.
///
/// Returns 0 for empty or zero-magnitude inputs so degenerate vectors don't
/// crash ranking. For L2-normalized embeddings this equals the dot product,
/// but normalizing here keeps the helper correct for arbitrary inputs and
/// independently unit-testable with fixed vectors.
func cosineSimilarity(_ a: [Float], _ b: [Float]) -> Float {
    let n = min(a.count, b.count)
    guard n > 0 else { return 0 }
    var dot: Float = 0
    var normA: Float = 0
    var normB: Float = 0
    for i in 0..<n {
        dot += a[i] * b[i]
        normA += a[i] * a[i]
        normB += b[i] * b[i]
    }
    let denom = normA.squareRoot() * normB.squareRoot()
    guard denom > 0 else { return 0 }
    return dot / denom
}

/// Bi-encoder rerank: score each document embedding against the query
/// embedding by cosine similarity, then return `(index, score)` pairs sorted
/// by descending score. When `topN` is provided (and in range) the result is
/// truncated to the top `topN` entries.
///
/// NOTE: This is a bi-encoder *approximation* of reranking — it reuses the
/// embedding model + cosine similarity, scoring the query and each document
/// independently. A true cross-encoder reranker (which scores every
/// query-document pair jointly) is a from-scratch follow-up; no MLX checkout
/// currently ships one.
func rerankByCosine(
    query: [Float],
    documents: [[Float]],
    topN: Int? = nil
) -> [(index: Int, score: Float)] {
    let scored = documents.enumerated().map { index, doc in
        (index: index, score: cosineSimilarity(query, doc))
    }
    let ranked = scored.sorted { $0.score > $1.score }
    if let topN, topN >= 0, topN < ranked.count {
        return Array(ranked.prefix(topN))
    }
    return ranked
}
