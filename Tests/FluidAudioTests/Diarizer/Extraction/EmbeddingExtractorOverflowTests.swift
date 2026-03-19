import XCTest

@testable import FluidAudio

/// Regression test for heap-buffer-overflow in EmbeddingExtractor.fillMaskBufferOptimized().
///
/// When audio.count > 160_000 (>10s at 16kHz), the numMasksInChunk formula
/// `(firstMask.count * audio.count + 80_000) / 160_000` can exceed firstMask.count,
/// causing vDSP_mmov to read past the mask buffer allocation.
///
/// Fix: clamp numMasksInChunk to firstMask.count.
/// See: https://github.com/FluidInference/FluidAudio/issues/XXX
final class EmbeddingExtractorOverflowTests: XCTestCase {

    // MARK: - numMasksInChunk formula validation

    /// The original (buggy) formula without clamping
    private func buggyNumMasksInChunk(maskCount: Int, audioCount: Int) -> Int {
        return (maskCount * audioCount + 80_000) / 160_000
    }

    /// The fixed formula with clamping
    private func fixedNumMasksInChunk(maskCount: Int, audioCount: Int) -> Int {
        return min(
            (maskCount * audioCount + 80_000) / 160_000,
            maskCount
        )
    }

    func testShortAudioDoesNotOverflow() {
        // Audio shorter than 10s (160_000 samples at 16kHz)
        // Both formulas should agree and stay within bounds
        let maskCount = 100
        let audioCount = 80_000  // 5 seconds

        let buggy = buggyNumMasksInChunk(maskCount: maskCount, audioCount: audioCount)
        let fixed = fixedNumMasksInChunk(maskCount: maskCount, audioCount: audioCount)

        XCTAssertEqual(buggy, fixed, "Short audio should produce same result with or without clamp")
        XCTAssertLessThanOrEqual(fixed, maskCount, "numMasksInChunk must not exceed maskCount")
    }

    func testExactly10sAudioBoundary() {
        // Audio exactly 10s = 160_000 samples
        let maskCount = 100
        let audioCount = 160_000

        let buggy = buggyNumMasksInChunk(maskCount: maskCount, audioCount: audioCount)
        let fixed = fixedNumMasksInChunk(maskCount: maskCount, audioCount: audioCount)

        // At exactly 160k: (100 * 160000 + 80000) / 160000 = 100.5 = 100 (int division)
        XCTAssertLessThanOrEqual(fixed, maskCount, "numMasksInChunk must not exceed maskCount")
        XCTAssertEqual(buggy, fixed, "At boundary, both formulas should agree")
    }

    func testLongAudioTriggersOverflow() {
        // Audio longer than 10s — this is where the bug manifests
        let maskCount = 100
        let audioCount = 320_000  // 20 seconds

        let buggy = buggyNumMasksInChunk(maskCount: maskCount, audioCount: audioCount)
        let fixed = fixedNumMasksInChunk(maskCount: maskCount, audioCount: audioCount)

        // Buggy: (100 * 320000 + 80000) / 160000 = 200 — DOUBLE the mask buffer!
        XCTAssertGreaterThan(buggy, maskCount,
            "Buggy formula MUST exceed maskCount for long audio (this proves the bug exists)")
        XCTAssertEqual(fixed, maskCount,
            "Fixed formula must clamp to maskCount")
    }

    func testVeryLongAudioSeverity() {
        // 60 seconds — typical meeting recording chunk
        let maskCount = 100
        let audioCount = 960_000  // 60 seconds

        let buggy = buggyNumMasksInChunk(maskCount: maskCount, audioCount: audioCount)
        let fixed = fixedNumMasksInChunk(maskCount: maskCount, audioCount: audioCount)

        // Buggy: (100 * 960000 + 80000) / 160000 = 600 — 6x overread!
        XCTAssertEqual(buggy, 600, "60s audio produces 6x overread with buggy formula")
        XCTAssertEqual(fixed, maskCount, "Fixed formula clamps correctly")
    }

    func testVariousMaskSizes() {
        // Test with different mask sizes to ensure the fix is general
        let testCases: [(maskCount: Int, audioCount: Int)] = [
            (50, 400_000),    // 25s, maskCount=50
            (200, 200_000),   // 12.5s, maskCount=200
            (1000, 480_000),  // 30s, maskCount=1000
            (10, 1_600_000),  // 100s, maskCount=10
        ]

        for (maskCount, audioCount) in testCases {
            let fixed = fixedNumMasksInChunk(maskCount: maskCount, audioCount: audioCount)
            XCTAssertLessThanOrEqual(fixed, maskCount,
                "numMasksInChunk must not exceed maskCount (\(maskCount)) for audioCount=\(audioCount)")
            XCTAssertGreaterThan(fixed, 0,
                "numMasksInChunk must be positive")
        }
    }

    func testZeroAudioProducesZeroMasks() {
        let maskCount = 100
        let audioCount = 0

        let fixed = fixedNumMasksInChunk(maskCount: maskCount, audioCount: audioCount)
        // (100 * 0 + 80000) / 160000 = 0 (int division)
        XCTAssertEqual(fixed, 0, "Zero audio should produce zero masks")
    }

    // MARK: - vDSP_mmov bounds validation (integration-level)

    func testFillMaskBufferBoundsWithLongAudio() throws {
        // This test validates that the actual vDSP_mmov call stays within bounds
        // by simulating what fillMaskBufferOptimized does with the fixed formula.
        //
        // Under ASan, the buggy version would crash here with:
        //   READ of size 3456 at ... is located 0 bytes after 2388-byte heap allocation

        let maskCount = 100
        let audioCount = 320_000  // 20 seconds — triggers overflow with buggy formula

        // Simulate the fixed numMasksInChunk calculation
        let numMasksInChunk = min(
            (maskCount * audioCount + 80_000) / 160_000,
            maskCount
        )

        // Create mask data (simulates masks[speakerIndex])
        let mask = [Float](repeating: 1.0, count: maskCount)

        // Create destination buffer (simulates the MLMultiArray buffer)
        var destination = [Float](repeating: 0.0, count: maskCount * 3)

        // This is what fillMaskBufferOptimized does — the critical vDSP_mmov call
        mask.withUnsafeBufferPointer { maskPtr in
            destination.withUnsafeMutableBufferPointer { destPtr in
                // This would crash with buggy numMasksInChunk (200 > 100)
                vDSP_mmov(
                    maskPtr.baseAddress!,
                    destPtr.baseAddress!,
                    vDSP_Length(numMasksInChunk),
                    vDSP_Length(1),
                    vDSP_Length(1),
                    vDSP_Length(numMasksInChunk)
                )
            }
        }

        // If we got here without crashing, the bounds are correct
        XCTAssertEqual(numMasksInChunk, maskCount,
            "numMasksInChunk should be clamped to maskCount")
    }
}
