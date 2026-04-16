import Testing
@testable import MacMLXCore

@Test
func versionStringIsSemverShape() {
    let version = MacMLXCore.version
    #expect(!version.isEmpty)
    let parts = version.split(separator: ".")
    #expect(parts.count == 3, "expected MAJOR.MINOR.PATCH, got \(version)")
    for part in parts {
        #expect(Int(part) != nil, "part \(part) is not numeric")
    }
}

@Test
func nameIsExpected() {
    #expect(MacMLXCore.name == "MacMLXCore")
}
