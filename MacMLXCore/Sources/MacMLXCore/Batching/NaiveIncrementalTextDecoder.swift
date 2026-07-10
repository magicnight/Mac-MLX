// Copyright © 2026 macMLX. English comments only.

import MLXLMCommon

/// Production ``IncrementalTextDecoder`` backing each batched-decode slot with
/// upstream's `NaiveStreamingDetokenizer` — the exact detokenizer the single
/// stream path uses (`MLXSwiftEngine.runLLMGeneration`), so batched per-slot
/// text matches sequential decode byte-for-byte.
///
/// `NaiveStreamingDetokenizer.next()` returns `nil` while a multi-token Unicode
/// scalar is still incomplete; this adapter maps that to an empty string so the
/// slot's stop-string filter always sees a well-defined chunk.
struct NaiveIncrementalTextDecoder: IncrementalTextDecoder {
    private var inner: NaiveStreamingDetokenizer

    init(tokenizer: any Tokenizer) {
        self.inner = NaiveStreamingDetokenizer(tokenizer: tokenizer)
    }

    mutating func decode(_ token: Int) -> String {
        inner.append(token: token)
        return inner.next() ?? ""
    }
}
