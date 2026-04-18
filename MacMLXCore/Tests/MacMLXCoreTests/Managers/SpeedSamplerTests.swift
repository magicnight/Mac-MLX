import XCTest
@testable import MacMLXCore

final class SpeedSamplerTests: XCTestCase {

    /// Drives a fake monotonic clock so tests don't depend on wall time.
    private final class FakeClock: @unchecked Sendable {
        private let lock = NSLock()
        private var now = Date(timeIntervalSince1970: 1_000_000)
        func advance(by seconds: TimeInterval) {
            lock.lock(); now = now.addingTimeInterval(seconds); lock.unlock()
        }
        func callAsFunction() -> Date {
            lock.lock(); defer { lock.unlock() }
            return now
        }
    }

    func testFirstSampleReturnsZero() {
        let clock = FakeClock()
        let sampler = SpeedSampler(clock: { clock() })
        XCTAssertEqual(sampler.record(bytes: 0), 0, accuracy: 0.001)
    }

    func testSubWindowSamplesHoldPreviousValue() {
        // Feed a sample, then many sub-500ms pulses. EMA should not move.
        let clock = FakeClock()
        let sampler = SpeedSampler(
            alpha: 0.15, minSampleInterval: 0.5,
            clock: { clock() }
        )
        _ = sampler.record(bytes: 0)
        clock.advance(by: 1.0)
        let first = sampler.record(bytes: 1_000_000)  // 1 MB over 1s -> 1 MB/s
        XCTAssertEqual(first, 1_000_000, accuracy: 1)

        // Next 10 callbacks inside a 100ms burst should all return `first`
        for _ in 0..<10 {
            clock.advance(by: 0.01)
            let held = sampler.record(bytes: 2_000_000)
            XCTAssertEqual(held, first, accuracy: 0.001,
                           "EMA must not update inside throttle window")
        }

        // After the full window elapses, EMA updates.
        clock.advance(by: 0.6)
        let updated = sampler.record(bytes: 3_000_000)
        XCTAssertGreaterThan(updated, 0)
    }

    func testEMAConvergesToNewRate() {
        // Seed with one steady rate, then jump to a new rate. The EMA
        // must lag (proving smoothing is actually happening, not just
        // arithmetic) and eventually converge.
        let clock = FakeClock()
        let sampler = SpeedSampler(
            alpha: 0.15, minSampleInterval: 0.5,
            clock: { clock() }
        )
        // Warm up with 5 MB/s steady rate — latches EMA at 5 MB/s.
        _ = sampler.record(bytes: 0)
        var bytes: Int64 = 0
        for _ in 0..<5 {
            clock.advance(by: 1.0)
            bytes += 5_000_000
            _ = sampler.record(bytes: bytes)
        }
        // Now jump to 10 MB/s. EMA should drift UP but lag due to alpha=0.15.
        var latest: Double = 0
        for _ in 0..<3 {
            clock.advance(by: 1.0)
            bytes += 10_000_000
            latest = sampler.record(bytes: bytes)
            // After N steps in the new regime, EMA ~ 10*(1-0.85^(N+1)) + 5*0.85^(N+1).
            // Step 1: ema ~ 5.75 MB/s. Step 3: ema ~ 6.91 MB/s. Far from 10 MB/s.
            XCTAssertLessThan(latest, 10_000_000,
                              "EMA must lag, not equal, the new rate")
            XCTAssertGreaterThan(latest, 5_000_000,
                                 "EMA must have moved toward the new rate")
        }
        // After many samples, it should be much closer to 10 MB/s.
        for _ in 0..<30 {
            clock.advance(by: 1.0)
            bytes += 10_000_000
            latest = sampler.record(bytes: bytes)
        }
        XCTAssertEqual(latest, 10_000_000, accuracy: 200_000,
                       "After prolonged steady rate, EMA converges")
    }

    func testNegativeBytesIgnored() {
        // Guard against impossible dbytes < 0 (shouldn't happen, but...)
        let clock = FakeClock()
        let sampler = SpeedSampler(clock: { clock() })
        _ = sampler.record(bytes: 1000)
        clock.advance(by: 1.0)
        let first = sampler.record(bytes: 2000)
        clock.advance(by: 1.0)
        let decreased = sampler.record(bytes: 500)
        XCTAssertEqual(decreased, first, accuracy: 0.001,
                       "Decreasing totals must not clobber EMA")
    }
}
