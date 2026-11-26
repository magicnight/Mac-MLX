import Dispatch
import Foundation
import Testing

/// `MLXTestActor` and `BlockingTestActor` keep their test bodies off the
/// cooperative pool, on threads of their own at utility QoS. These tests check
/// that each actor runs its bodies where it says, and that no MLX-gated test
/// goes without one.
@Suite("Off-pool test actors")
struct OffPoolTestActorTests {

    /// The label of the queue the caller is running on.
    private static func currentQueueLabel() -> String {
        String(cString: __dispatch_queue_get_label(nil))
    }

    /// The name of the thread the caller is running on.
    private static func currentThreadName() -> String {
        Thread.current.name ?? ""
    }

    /// Whether the caller runs at utility QoS, below the tests' own priority.
    private static func runsAtUtility() -> Bool {
        qos_class_self() == QOS_CLASS_UTILITY
    }

    /// The control: without an actor, a test body runs on the cooperative pool,
    /// so the label below tells the two places apart.
    @Test
    func aTestWithoutAnActorRunsOnTheCooperativePool() async {
        #expect(Self.currentQueueLabel().hasSuffix(".cooperative"))
    }

    @Test @MLXTestActor
    func anMLXTestRunsOnTheMLXThreadAtUtility() {
        #expect(Self.currentThreadName() == MLXTestActor.threadName)
        #expect(Self.runsAtUtility())
    }

    /// An async body comes back to the thread after it suspends.
    @Test @MLXTestActor
    func anAsyncMLXTestResumesOnTheMLXThread() async throws {
        try await Task.sleep(for: .milliseconds(1))
        #expect(Self.currentThreadName() == MLXTestActor.threadName)
        #expect(Self.runsAtUtility())
    }

    @Test @BlockingTestActor
    func aBlockingTestRunsOnItsOwnThreadAtUtility() {
        #expect(Self.currentThreadName() == BlockingTestActor.threadName)
        #expect(Self.runsAtUtility())
    }

    /// Every suite or test that carries the MLX gate is isolated to
    /// `MLXTestActor`. One without it puts synchronous Metal work back on the
    /// pool, where it holds every other test of the run.
    @Test
    func everyMLXGatedDeclarationIsIsolatedToTheMLXActor() throws {
        let testsRoot = URL(filePath: #filePath)
            .deletingLastPathComponent()  // Helpers
            .deletingLastPathComponent()  // MacMLXCoreTests
        let files = try #require(
            FileManager.default.enumerator(at: testsRoot, includingPropertiesForKeys: nil),
            "cannot list \(testsRoot.path)")
        let gate = /\.enabled\(\s*if:\s*mlxMetallibIsAvailable/
        let declaration = /\b(struct|class|actor|enum|extension|func)\s+\w/

        var gated = 0
        var missing: [String] = []
        for case let file as URL in files where file.pathExtension == "swift" {
            let lines = try String(contentsOf: file, encoding: .utf8)
                .split(separator: "\n", omittingEmptySubsequences: false)
            // The attributes since the last declaration: `@MLXTestActor` may
            // come before or after the gate.
            var gateLine: Int?
            var isolated = false
            for (index, line) in lines.enumerated() {
                // Code only: no line comment, no string literal.
                let code = (line.split(separator: "//", maxSplits: 1, omittingEmptySubsequences: false).first ?? "")
                    .replacing(/"[^"]*"/, with: "\"\"")
                if code.contains(gate) { gateLine = index }
                if code.contains("@MLXTestActor") { isolated = true }
                if code.contains(declaration) {
                    if let start = gateLine {
                        gated += 1
                        if !isolated { missing.append("\(file.lastPathComponent):\(start + 1)") }
                    }
                    gateLine = nil
                    isolated = false
                }
            }
        }

        try #require(gated > 0, "found no MLX-gated declaration under \(testsRoot.path)")
        #expect(
            missing.isEmpty,
            """
            MLX-gated without @MLXTestActor: \(missing.joined(separator: ", ")). \
            Isolate each to MLXTestActor (see its doc comment) so its Metal work \
            runs off the cooperative pool.
            """)
    }
}
