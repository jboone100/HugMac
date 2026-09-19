import Foundation
import LocalLabCore

/// `locallab-bench --install <repo>` — install (or adopt, or resume) a model into the default
/// library, printing progress. Repos an engine knows use its component manifest;
/// everything else uses the heuristics.
enum InstallCommand {
    static func run(arguments: [String]) async throws {
        guard let repo = arguments.first else {
            throw Failure("usage: locallab-bench --install <org/repo>")
        }
        let manifest = ComponentManifest.known(for: repo)
        let installer = ModelInstaller()
        let staged = await installer.stagedBytes(repo)
        print("installing \(repo) into \(installer.store.baseDirectory.path)")
        if staged > 0 {
            print(String(format: "  %.1f MB already staged — resuming", Double(staged) / 1e6))
        }

        let started = Date()
        let last = LastPrinted()
        let model = try await installer.install(repo, manifest: manifest) { progress in
            let line = String(
                format: "  %-11@ %5.1f%%  %8.1f / %.1f MB  %@",
                progress.phase.rawValue as NSString, progress.fraction * 100,
                Double(progress.bytesDone) / 1e6, Double(progress.bytesTotal) / 1e6,
                progress.file as NSString
            )
            if last.shouldPrint(phase: progress.phase.rawValue, file: progress.file) { print(line) }
        }
        print(String(format: "installed %@ @ %@ — %.2f GB, %d files, %.1f s",
                     model.repo as NSString, String(model.revision.prefix(12)) as NSString,
                     Double(model.sizeBytes) / 1e9, model.files.count,
                     Date().timeIntervalSince(started)))
        for file in model.files {
            print("  \(file.path)  \(file.checksum.map { String($0.prefix(24)) + "…" } ?? "—")")
        }
    }

    final class LastPrinted: @unchecked Sendable {
        private let lock = NSLock()
        private var key = ""
        private var at = Date.distantPast
        func shouldPrint(phase: String, file: String) -> Bool {
            lock.withLock {
                let newKey = phase + file
                let now = Date()
                guard newKey != key || now.timeIntervalSince(at) > 2 else { return false }
                key = newKey
                at = now
                return true
            }
        }
    }
}
