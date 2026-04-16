import Testing
import Foundation
@testable import MacMLXCore

@Test
func engineIDRawValuesMatchSpec() {
    #expect(EngineID.mlxSwift.rawValue == "mlx-swift-lm")
    #expect(EngineID.swiftLM.rawValue == "swift-lm")
    #expect(EngineID.pythonMLX.rawValue == "python-mlx")
}

@Test
func engineIDRoundTripsThroughJSON() throws {
    for id in EngineID.allCases {
        let data = try JSONEncoder().encode(id)
        let decoded = try JSONDecoder().decode(EngineID.self, from: data)
        #expect(decoded == id)
    }
}

@Test
func engineIDIsCaseIterable() {
    #expect(EngineID.allCases.count == 3)
}
