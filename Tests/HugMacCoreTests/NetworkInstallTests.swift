import Foundation
import Testing
@testable import HugMacCore

/// Against the real Hugging Face, because the fake can't prove the part most likely to break:
/// that a `Range` request survives the redirect from huggingface.co to its CDN, so a resumed
/// download really resumes. Opt-in — it downloads ~350 MB:
///
///     HUGMAC_NETWORK_TESTS=1 swift test --filter NetworkInstallTests
@Suite(
    "Network installs",
    .enabled(if: ProcessInfo.processInfo.environment["HUGMAC_NETWORK_TESTS"] == "1")
)
struct NetworkInstallTests {

    final class FirstValue: @unchecked Sendable {
        private let lock = NSLock()
        private var value: Int64?
        func setIfUnset(_ newValue: Int64) { lock.withLock { if value == nil { value = newValue } } }
        var result: Int64? { lock.withLock { value } }
    }

    @Test("A cancelled real download resumes across the CDN redirect and verifies")
    func resumesRealDownload() async throws {
        let store = temporaryStore()
        defer { try? FileManager.default.removeItem(at: store.baseDirectory) }
        let installer = ModelInstaller(store: store)
        let repo = "mlx-community/Qwen3-0.6B-4bit"
        let partial = store.stagingDirectory(forRepo: repo)
            .appendingPathComponent("model.safetensors.partial")

        // Start, let ~60 MB of the 335 MB weight file land, then pull the plug.
        let first = Task { try await installer.install(repo) }
        let deadline = Date().addingTimeInterval(180)
        while Integrity.fileSize(partial) < 60_000_000, Date() < deadline {
            try await Task.sleep(for: .milliseconds(50))
        }
        await installer.cancel(repo)
        _ = try? await first.value
        let before = Integrity.fileSize(partial)
        #expect(before >= 60_000_000, "expected a substantial partial, got \(before) bytes")
        #expect(!store.isInstalled(repo: repo))

        // Resume. If the Range header were lost on redirect, the server would answer 200 and
        // the first progress report for the weight file would restart near zero.
        let firstReport = FirstValue()
        let model = try await installer.install(repo) { progress in
            if progress.phase == .downloading, progress.file == "model.safetensors" {
                firstReport.setIfUnset(progress.bytesDone)
            }
        }
        let resumedAt = try #require(firstReport.result)
        #expect(resumedAt >= before, "resumed at \(resumedAt), partial was \(before)")

        #expect(store.isInstalled(repo: repo))
        #expect(model.files.contains { $0.path == "model.safetensors" && $0.checksum?.hasPrefix("sha256:") == true })
        print("resume: partial \(before) bytes → first report \(resumedAt) bytes; revision \(model.revision)")
    }
}
