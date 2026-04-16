import Testing
@testable import MacMLXCore

@Test
func memoryProbeReportsPositiveTotalRam() {
    let gb = MemoryProbe.totalMemoryGB()
    #expect(gb > 0, "expected positive RAM, got \(gb)")
    // Sanity: any modern Apple Silicon Mac has >=4 GB and <=2 TB.
    #expect(gb >= 4)
    #expect(gb <= 2048)
}

@Test
func memoryProbeReturnsConsistentValue() {
    let a = MemoryProbe.totalMemoryGB()
    let b = MemoryProbe.totalMemoryGB()
    #expect(a == b)
}
