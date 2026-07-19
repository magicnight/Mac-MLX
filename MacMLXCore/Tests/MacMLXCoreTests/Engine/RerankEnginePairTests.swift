import Testing
@testable import MacMLXCore

/// Pure unit tests for `RerankEngine.buildPair` — the parity-critical BERT
/// cross-encoder pair assembly. A stub `encode` (fixed ids per string) stands
/// in for a real tokenizer, so these need NO model download and NO Metal.
///
/// What's asserted here is the wire format a real checkpoint depends on:
/// `[CLS] query [SEP] document [SEP]`, segment ids 0 over the query span and 1
/// over the document span, and HF-default `longest_first` truncation. Numeric
/// parity against a real tokenizer is deferred (needs a download).
@Suite("RerankEngine pair assembly")
struct RerankEnginePairTests {

    /// [CLS]=101, [SEP]=102 (BERT's actual ids), generous maxLength so nothing
    /// truncates. `encode` returns fixed ids per string.
    private func build(
        query: String, document: String,
        encode: @escaping (String) -> [Int], maxLength: Int = 512
    ) -> (ids: [Int], tokenTypeIds: [Int]) {
        RerankEngine.buildPair(
            query: query, document: document,
            clsId: 101, sepId: 102, maxLength: maxLength, encode: encode)
    }

    @Test
    func assemblesClsQuerySepDocSepWithSegmentIds() {
        let ids: [String: [Int]] = ["q": [10, 11], "d": [20, 21, 22]]
        let pair = build(query: "q", document: "d", encode: { ids[$0] ?? [] })

        // [CLS] 10 11 [SEP] 20 21 22 [SEP]
        #expect(pair.ids == [101, 10, 11, 102, 20, 21, 22, 102])
        // 0 over [CLS] query [SEP] (4 tokens), 1 over document [SEP] (4 tokens).
        #expect(pair.tokenTypeIds == [0, 0, 0, 0, 1, 1, 1, 1])
        // ids and segment ids are always the same length.
        #expect(pair.ids.count == pair.tokenTypeIds.count)
    }

    @Test
    func segmentBoundaryFallsAfterQuerySep() {
        let ids: [String: [Int]] = ["q": [10], "d": [20, 21, 22, 23]]
        let pair = build(query: "q", document: "d", encode: { ids[$0] ?? [] })
        let zeros = pair.tokenTypeIds.filter { $0 == 0 }.count
        let ones = pair.tokenTypeIds.filter { $0 == 1 }.count
        #expect(zeros == 1 + 1 + 1)   // [CLS] + 1 query token + [SEP]
        #expect(ones == 4 + 1)        // 4 document tokens + [SEP]
    }

    @Test
    func emptyQueryStillWellFormed() {
        let ids: [String: [Int]] = ["": [], "d": [20, 21]]
        let pair = build(query: "", document: "d", encode: { ids[$0] ?? [] })
        #expect(pair.ids == [101, 102, 20, 21, 102])
        #expect(pair.tokenTypeIds == [0, 0, 1, 1, 1])
    }

    @Test
    func longestFirstTruncationTrimsTheLongerSegment() {
        // maxLength 6 → content budget 3 (reserving [CLS] + [SEP] + [SEP]).
        // query 4 tokens, document 1 → query is longer, trimmed first from the
        // right until query(2)+document(1) == 3.
        let ids: [String: [Int]] = ["q": [10, 11, 12, 13], "d": [20]]
        let pair = build(query: "q", document: "d", encode: { ids[$0] ?? [] }, maxLength: 6)
        #expect(pair.ids == [101, 10, 11, 102, 20, 102])
        #expect(pair.ids.count == 6)
        #expect(pair.tokenTypeIds == [0, 0, 0, 0, 1, 1])
    }

    @Test
    func longestFirstTruncationBreaksTiesByTrimmingDocument() {
        // Equal-length segments (2 and 2), budget 3 → tie trims the document
        // (the second sequence), matching HF transformers.
        let ids: [String: [Int]] = ["q": [10, 11], "d": [20, 21]]
        let pair = build(query: "q", document: "d", encode: { ids[$0] ?? [] }, maxLength: 6)
        // query kept intact (2), document trimmed to 1.
        #expect(pair.ids == [101, 10, 11, 102, 20, 102])
    }

    @Test
    func noTruncationWhenWithinBudget() {
        let ids: [String: [Int]] = ["q": [10], "d": [20]]
        let pair = build(query: "q", document: "d", encode: { ids[$0] ?? [] }, maxLength: 512)
        #expect(pair.ids == [101, 10, 102, 20, 102])
    }
}
