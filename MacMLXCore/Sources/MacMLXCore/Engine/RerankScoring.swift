import Foundation

// MARK: - Shared ranking

/// Rank per-document scores descending and truncate to `topN`.
///
/// The ranking core shared by BOTH `/v1/rerank` paths — the true
/// cross-encoder (``RerankEngine`` raw logits) and the bi-encoder cosine
/// fallback (`rerankByCosine`) — so they order and truncate identically and
/// this logic is unit-testable in isolation from either scorer.
///
/// `scores[i]` is document `i`'s relevance; the result is `(index, score)`
/// pairs sorted by descending score. `sorted(by:)` is not guaranteed stable,
/// so ties may order arbitrarily (acceptable — equal scores are equally
/// relevant). When `topN` is provided and in range the result is truncated to
/// the top `topN`; a negative or out-of-range `topN` returns the full ranking.
func rankAndTruncate(scores: [Float], topN: Int? = nil) -> [(index: Int, score: Float)] {
    let ranked = scores.enumerated()
        .map { (index: $0.offset, score: $0.element) }
        .sorted { $0.score > $1.score }
    if let topN, topN >= 0, topN < ranked.count {
        return Array(ranked.prefix(topN))
    }
    return ranked
}

/// Logistic sigmoid mapping a raw cross-encoder relevance logit to `(0, 1)`.
///
/// Exposed as the API `relevance_score` for the cross-encoder path so callers
/// get a bounded, Cohere/Jina-style score. Being strictly monotonic, it NEVER
/// reorders what `rankAndTruncate` produced from the raw logits — it only
/// rescales for display. Computed in `Double` for endpoint JSON.
func rerankSigmoid(_ logit: Float) -> Double {
    1.0 / (1.0 + Foundation.exp(-Double(logit)))
}

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
    let scores = documents.map { cosineSimilarity(query, $0) }
    return rankAndTruncate(scores: scores, topN: topN)
}
