import Foundation
import Testing

@testable import MacMLXCore

@Suite("ReasoningCarrier")
struct ReasoningCarrierTests {

    @Test
    func roundTrips() {
        for text in ["Plan the call.", "多行\n推理 with emoji \u{1F9E0} and \"quotes\"", String(repeating: "x", count: 10_000)] {
            let blob = ReasoningCarrier.encode(text)
            #expect(blob.hasPrefix("macmlx.r1."))
            #expect(ReasoningCarrier.decode(blob) == text)
        }
    }

    /// Only macMLX's own blobs decode: a real provider's signature or
    /// encrypted content, even valid base64, is not reasoning macMLX can read.
    @Test
    func rejectsForeignAndInvalidBlobs() {
        let foreignBase64 = Data("Plan the call.".utf8).base64EncodedString()
        #expect(ReasoningCarrier.decode(foreignBase64) == nil)
        #expect(ReasoningCarrier.decode("macmlx") == nil)
        #expect(ReasoningCarrier.decode("gAAAAABo-opaque") == nil)
        #expect(ReasoningCarrier.decode("macmlx.r2." + foreignBase64) == nil)
        #expect(ReasoningCarrier.decode("macmlx.r1.not base64!") == nil)
        // Valid base64 whose bytes are not UTF-8.
        #expect(ReasoningCarrier.decode("macmlx.r1." + Data([0xFF, 0xFE, 0xFD]).base64EncodedString()) == nil)
        #expect(ReasoningCarrier.decode("macmlx.r1.") == nil)
        #expect(ReasoningCarrier.decode("") == nil)
        #expect(ReasoningCarrier.decode(nil) == nil)
    }
}
