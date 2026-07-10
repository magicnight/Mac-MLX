import Testing

@testable import MacMLXCore

// MARK: - TokenVocabularyTable Tests (Track C — C1)
//
// Pure, MLX-free classification of a scripted vocabulary. No tokenizer, no
// Metal — the decode closure stands in for `tokenizer.decode(tokenIds:[id])`.

@Suite("TokenVocabularyTable")
struct TokenVocabularyTableTests {

    @Test
    func classifiesNormalTokensAsTheirBytes() {
        let vocab = ["{", "\"", "true", " hello", "123"]
        let table = TokenVocabularyTable(
            vocabularySize: vocab.count,
            stopTokenIDs: [],
            decode: { vocab[$0] }
        )
        #expect(table.classification(of: 0) == .bytes(Array("{".utf8)))
        #expect(table.classification(of: 3) == .bytes(Array(" hello".utf8)))
        #expect(table.classification(of: 4) == .bytes(Array("123".utf8)))
    }

    @Test
    func classifiesStopTokensAsEOS() {
        let vocab = ["a", "</s>", "b"]
        let table = TokenVocabularyTable(
            vocabularySize: vocab.count,
            stopTokenIDs: [1],
            decode: { vocab[$0] }
        )
        #expect(table.classification(of: 1) == .eos)
        // A stop id wins even if it decodes to ordinary-looking text.
        #expect(table.classification(of: 0) == .bytes(Array("a".utf8)))
    }

    @Test
    func classifiesEmptyAndReplacementTokensAsUnusable() {
        // id 1 decodes to nil, id 2 to empty, id 3 contains U+FFFD (a
        // byte-fragment token whose exact bytes are unrecoverable).
        let table = TokenVocabularyTable(
            vocabularySize: 4,
            stopTokenIDs: [],
            decode: { id in
                switch id {
                case 0: return "ok"
                case 1: return nil
                case 2: return ""
                default: return "x\u{FFFD}"
                }
            }
        )
        #expect(table.classification(of: 0) == .bytes(Array("ok".utf8)))
        #expect(table.classification(of: 1) == .unusable)
        #expect(table.classification(of: 2) == .unusable)
        #expect(table.classification(of: 3) == .unusable)
    }

    @Test
    func outOfRangeIsUnusable() {
        let table = TokenVocabularyTable(vocabularySize: 2, stopTokenIDs: [], decode: { _ in "a" })
        #expect(table.count == 2)
        #expect(table.classification(of: -1) == .unusable)
        #expect(table.classification(of: 99) == .unusable)
    }
}
