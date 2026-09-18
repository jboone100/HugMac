import CryptoKit
import Foundation
import Testing
@testable import HugMacCore

// MARK: - A fake Hugging Face

/// Serves in-memory repos, and can interrupt or corrupt a download on demand — the failure
/// modes a multi-gigabyte install actually meets.
final class FakeHub: HubClient, @unchecked Sendable {
    struct Call: Equatable {
        let path: String
        let startOffset: Int64
    }

    private let lock = NSLock()
    private var repos: [String: (revision: String, files: [String: Data], gated: Bool)] = [:]
    private var _calls: [Call] = []
    /// Write this many bytes of the file, then fail — a dropped connection.
    var interruptAfter: [String: Int] = [:]
    /// Serve wrong bytes for this file — a corrupted transfer.
    var corrupt: Set<String> = []

    var calls: [Call] { lock.withLock { _calls } }

    func add(repo: String, revision: String = "abc123", files: [String: Data], gated: Bool = false) {
        lock.withLock { repos[repo] = (revision, files, gated) }
    }

    func snapshot(repo: String) async throws -> RepoSnapshot {
        guard let entry = lock.withLock({ repos[repo] }) else { throw HubError.notFound(repo: repo) }
        if entry.gated { throw HubError.needsAuth(repo: repo) }
        let files = entry.files.map { path, data -> RemoteFile in
            // Weights behave like LFS (SHA-256); everything else like git blobs (SHA-1).
            if path.hasSuffix(".safetensors") {
                let hex = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
                return RemoteFile(path: path, size: Int64(data.count), expected: .sha256(hex))
            }
            var hasher = Insecure.SHA1()
            hasher.update(data: Data("blob \(data.count)\u{0}".utf8))
            hasher.update(data: data)
            let hex = hasher.finalize().map { String(format: "%02x", $0) }.joined()
            return RemoteFile(path: path, size: Int64(data.count), expected: .gitBlobSHA1(hex))
        }
        return RepoSnapshot(repo: repo, revision: entry.revision, files: files)
    }

    func download(repo: String, revision: String, path: String, to destination: URL,
                  progress: @Sendable @escaping (Int64) -> Void) async throws {
        let (data, interrupt, isCorrupt) = lock.withLock {
            (repos[repo]?.files[path] ?? Data(), interruptAfter[path], corrupt.contains(path))
        }
        if !FileManager.default.fileExists(atPath: destination.path) {
            FileManager.default.createFile(atPath: destination.path, contents: nil)
        }
        let start = Integrity.fileSize(destination)
        lock.withLock { _calls.append(Call(path: path, startOffset: start)) }

        var body = data.subdata(in: Int(start) ..< data.count)
        if isCorrupt, !body.isEmpty { body[body.startIndex] ^= 0xFF }
        let handle = try FileHandle(forWritingTo: destination)
        defer { try? handle.close() }
        try handle.seekToEnd()
        if let interrupt, Int(start) < interrupt {
            try handle.write(contentsOf: body.prefix(interrupt - Int(start)))
            lock.withLock { interruptAfter[path] = nil }   // fail once
            throw URLError(.networkConnectionLost)
        }
        try handle.write(contentsOf: body)
        progress(Int64(data.count))
    }
}

// MARK: - Test fixtures

/// A structurally valid safetensors file holding F32 tensors of the given element counts.
func makeSafetensors(_ tensors: [(String, Int)]) -> Data {
    var header: [String: Any] = [:]
    var offset = 0
    for (name, count) in tensors {
        header[name] = ["dtype": "F32", "shape": [count], "data_offsets": [offset, offset + count * 4]]
        offset += count * 4
    }
    let json = (try? JSONSerialization.data(withJSONObject: header, options: [.sortedKeys])) ?? Data()
    var data = Data()
    var length = UInt64(json.count).littleEndian
    withUnsafeBytes(of: &length) { data.append(contentsOf: $0) }
    data.append(json)
    data.append(Data(repeating: 0x3F, count: offset))
    return data
}

func temporaryStore() -> ModelStore {
    let url = FileManager.default.temporaryDirectory
        .appendingPathComponent("hugmac-tests-\(UUID().uuidString)", isDirectory: true)
    return ModelStore(root: StorageRoot(url: url))
}

let plentyOfDisk: @Sendable () -> Int64 = { 1 << 40 }

/// The SeedVR2 repo's shape, at toy size.
func seedVR2Files() -> [String: Data] {
    [
        ".gitattributes": Data("*.safetensors filter=lfs".utf8),
        "README.md": Data("# SeedVR2".utf8),
        "config.json": Data(#"{"model_type":"seedvr2"}"#.utf8),
        "pos_emb.safetensors": makeSafetensors([("pos_emb", 64)]),
        "vae.safetensors": makeSafetensors([
            ("encoder.conv_in.weight", 128), ("decoder.conv_out.weight", 128),
        ]),
        "transformer.safetensors": makeSafetensors([
            ("vid_in.proj.weight", 4096), ("vid_out.proj.weight", 4096),
        ]),
    ]
}

let seedRepo = "mlx-community/SeedVR2-3B-mlx-int8"

// MARK: - Installer

@Suite("Model installer")
struct ModelInstallerTests {

    @Test("Installs exactly the manifest's files, writes the marker and registry, clears staging")
    func installsManifestFiles() async throws {
        let hub = FakeHub(); hub.add(repo: seedRepo, files: seedVR2Files())
        let store = temporaryStore()
        defer { try? FileManager.default.removeItem(at: store.baseDirectory) }
        let installer = ModelInstaller(store: store, hub: hub, availableBytes: plentyOfDisk)

        let model = try await installer.install(seedRepo, manifest: SeedVR2Variant.manifest)

        #expect(Set(model.files.map(\.path)) ==
                ["config.json", "pos_emb.safetensors", "transformer.safetensors", "vae.safetensors"])
        #expect(model.revision == "abc123")
        #expect(store.isInstalled(repo: seedRepo))
        #expect(!FileManager.default.fileExists(atPath: store.directory(forRepo: seedRepo)
            .appendingPathComponent("README.md").path))
        #expect(!FileManager.default.fileExists(atPath: store.stagingDirectory(forRepo: seedRepo).path))
        #expect(InstallRegistry.load(store).models[seedRepo] == model)
        #expect(model.files.allSatisfy { $0.checksum != nil }, "every file carries a checksum")
    }

    @Test("Heuristic selection still skips docs when no manifest is given")
    func heuristicSelection() async throws {
        let hub = FakeHub()
        hub.add(repo: "org/tiny", files: [
            "config.json": Data("{}".utf8),
            "model.safetensors": makeSafetensors([("w", 16)]),
            "README.md": Data("docs".utf8),
            "convert.py": Data("print()".utf8),
        ])
        let scratch = temporaryStore()
        defer { try? FileManager.default.removeItem(at: scratch.baseDirectory) }
        let installer = ModelInstaller(store: scratch, hub: hub, availableBytes: plentyOfDisk)
        let model = try await installer.install("org/tiny")
        #expect(Set(model.files.map(\.path)) == ["config.json", "model.safetensors"])
    }

    @Test("An interrupted download resumes from where it stopped")
    func resumesPartial() async throws {
        let hub = FakeHub(); hub.add(repo: seedRepo, files: seedVR2Files())
        let transformerSize = seedVR2Files()["transformer.safetensors"]?.count ?? 0
        hub.interruptAfter["transformer.safetensors"] = transformerSize / 2
        let store = temporaryStore()
        defer { try? FileManager.default.removeItem(at: store.baseDirectory) }
        let installer = ModelInstaller(store: store, hub: hub, availableBytes: plentyOfDisk)

        await #expect(throws: (any Error).self) {
            try await installer.install(seedRepo, manifest: SeedVR2Variant.manifest)
        }
        #expect(!store.isInstalled(repo: seedRepo))
        #expect(await installer.stagedBytes(seedRepo) > 0, "a failure keeps what was downloaded")

        try await installer.install(seedRepo, manifest: SeedVR2Variant.manifest)
        #expect(store.isInstalled(repo: seedRepo))
        let transformerCalls = hub.calls.filter { $0.path == "transformer.safetensors" }
        #expect(transformerCalls.map(\.startOffset) == [0, Int64(transformerSize / 2)],
                "the second attempt must resume, not restart")
        // Files that finished the first time are not fetched again.
        #expect(hub.calls.filter { $0.path == "vae.safetensors" }.count == 1)
    }

    @Test("A corrupted file fails its checksum, is discarded, and never gets a marker")
    func rejectsCorruption() async throws {
        let hub = FakeHub(); hub.add(repo: seedRepo, files: seedVR2Files())
        hub.corrupt = ["vae.safetensors"]
        let store = temporaryStore()
        defer { try? FileManager.default.removeItem(at: store.baseDirectory) }
        let installer = ModelInstaller(store: store, hub: hub, availableBytes: plentyOfDisk)

        do {
            try await installer.install(seedRepo, manifest: SeedVR2Variant.manifest)
            Issue.record("a corrupted file must not install")
        } catch let error as InstallError {
            guard case .verification = error else {
                Issue.record("expected a verification failure, got \(error)"); return
            }
        }
        #expect(!store.isInstalled(repo: seedRepo))
        let partial = store.stagingDirectory(forRepo: seedRepo)
            .appendingPathComponent("vae.safetensors.partial")
        #expect(!FileManager.default.fileExists(atPath: partial.path),
                "a file that failed its checksum is not resumable")

        hub.corrupt = []
        try await installer.install(seedRepo, manifest: SeedVR2Variant.manifest)
        #expect(store.isInstalled(repo: seedRepo))
    }

    @Test("A weight file missing a required tensor is refused at install, not at run time")
    func requiredTensors() async throws {
        var files = seedVR2Files()
        // The right name and a valid structure — but not the file the engine needs.
        files["vae.safetensors"] = makeSafetensors([("encoder.conv_in.weight", 128)])
        let hub = FakeHub(); hub.add(repo: seedRepo, files: files)
        let store = temporaryStore()
        defer { try? FileManager.default.removeItem(at: store.baseDirectory) }
        let installer = ModelInstaller(store: store, hub: hub, availableBytes: plentyOfDisk)

        do {
            try await installer.install(seedRepo, manifest: SeedVR2Variant.manifest)
            Issue.record("an incomplete component must not install")
        } catch let error as InstallError {
            guard case .incomplete(_, let file, let detail) = error else {
                Issue.record("expected .incomplete, got \(error)"); return
            }
            #expect(file == "vae.safetensors")
            #expect(detail.contains("decoder.conv_out.weight"))
        }
        #expect(!store.isInstalled(repo: seedRepo))
    }

    @Test("Files already in the model directory are verified and adopted, not re-downloaded")
    func adoptsExistingFiles() async throws {
        let hub = FakeHub(); hub.add(repo: seedRepo, files: seedVR2Files())
        let store = temporaryStore()
        defer { try? FileManager.default.removeItem(at: store.baseDirectory) }
        try store.prepare()
        // What a hand-made copy looks like: the files, no marker.
        let directory = store.directory(forRepo: seedRepo)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        for (path, data) in seedVR2Files() {
            try data.write(to: directory.appendingPathComponent(path))
        }
        let installer = ModelInstaller(store: store, hub: hub, availableBytes: plentyOfDisk)
        try await installer.install(seedRepo, manifest: SeedVR2Variant.manifest)

        #expect(store.isInstalled(repo: seedRepo))
        #expect(hub.calls.isEmpty, "every file verified in place — nothing to fetch")
    }

    @Test("An already-installed model makes no network calls")
    func alreadyInstalled() async throws {
        let hub = FakeHub(); hub.add(repo: seedRepo, files: seedVR2Files())
        let store = temporaryStore()
        defer { try? FileManager.default.removeItem(at: store.baseDirectory) }
        let installer = ModelInstaller(store: store, hub: hub, availableBytes: plentyOfDisk)
        try await installer.install(seedRepo, manifest: SeedVR2Variant.manifest)
        let callsAfterFirst = hub.calls.count
        try await installer.install(seedRepo, manifest: SeedVR2Variant.manifest)
        #expect(hub.calls.count == callsAfterFirst)
    }

    @Test("Not enough disk is refused before anything downloads, counting only what's left")
    func diskPreflight() async throws {
        let hub = FakeHub(); hub.add(repo: seedRepo, files: seedVR2Files())
        let scratch = temporaryStore()
        defer { try? FileManager.default.removeItem(at: scratch.baseDirectory) }
        let installer = ModelInstaller(store: scratch, hub: hub, availableBytes: { 1_000 })
        do {
            try await installer.install(seedRepo, manifest: SeedVR2Variant.manifest)
            Issue.record("should refuse for lack of disk")
        } catch let error as InstallError {
            guard case .insufficientDisk(let required, let available) = error else {
                Issue.record("expected .insufficientDisk, got \(error)"); return
            }
            #expect(available == 1_000)
            // No 2× factor: renaming into place needs no second copy.
            #expect(required < 2 * ModelInstaller.safetyBufferBytes)
        }
        #expect(hub.calls.isEmpty)
    }

    @Test("Two requests for the same repo share one download")
    func concurrentInstallsDedupe() async throws {
        let hub = FakeHub(); hub.add(repo: seedRepo, files: seedVR2Files())
        let store = temporaryStore()
        defer { try? FileManager.default.removeItem(at: store.baseDirectory) }
        let installer = ModelInstaller(store: store, hub: hub, availableBytes: plentyOfDisk)
        async let first = installer.install(seedRepo, manifest: SeedVR2Variant.manifest)
        async let second = installer.install(seedRepo, manifest: SeedVR2Variant.manifest)
        let (a, b) = try await (first, second)
        #expect(a.repo == b.repo)
        #expect(hub.calls.filter { $0.path == "transformer.safetensors" }.count == 1)
    }

    @Test("Uninstall removes the files and the registry entry")
    func uninstall() async throws {
        let hub = FakeHub(); hub.add(repo: seedRepo, files: seedVR2Files())
        let store = temporaryStore()
        defer { try? FileManager.default.removeItem(at: store.baseDirectory) }
        let installer = ModelInstaller(store: store, hub: hub, availableBytes: plentyOfDisk)
        try await installer.install(seedRepo, manifest: SeedVR2Variant.manifest)
        try await installer.uninstall(seedRepo)
        #expect(!FileManager.default.fileExists(atPath: store.directory(forRepo: seedRepo).path))
        #expect(InstallRegistry.load(store).models[seedRepo] == nil)
    }

    @Test("The registry drops entries whose files were deleted behind the app's back")
    func registryReconciles() async throws {
        let hub = FakeHub(); hub.add(repo: seedRepo, files: seedVR2Files())
        let store = temporaryStore()
        defer { try? FileManager.default.removeItem(at: store.baseDirectory) }
        let installer = ModelInstaller(store: store, hub: hub, availableBytes: plentyOfDisk)
        try await installer.install(seedRepo, manifest: SeedVR2Variant.manifest)
        try FileManager.default.removeItem(at: store.directory(forRepo: seedRepo))
        #expect(InstallRegistry.load(store).models.isEmpty)
    }

    @Test("A gated repo surfaces as needs-auth, so the UI can route to the token field")
    func gatedRepo() async throws {
        let hub = FakeHub(); hub.add(repo: "org/gated", files: ["config.json": Data("{}".utf8)], gated: true)
        let scratch = temporaryStore()
        defer { try? FileManager.default.removeItem(at: scratch.baseDirectory) }
        let installer = ModelInstaller(store: scratch, hub: hub, availableBytes: plentyOfDisk)
        await #expect(throws: HubError.needsAuth(repo: "org/gated")) {
            try await installer.install("org/gated")
        }
        #expect(HubError.fromStatus(401, repo: "x") == .needsAuth(repo: "x"))
        #expect(HubError.fromStatus(403, repo: "x") == .needsAuth(repo: "x"))
        #expect(HubError.fromStatus(404, repo: "x") == .notFound(repo: "x"))
        #expect(HubError.fromStatus(206, repo: "x") == nil)
    }
}

// MARK: - Integrity and headers

@Suite("Integrity")
struct IntegrityTests {

    @Test("Git blob SHA-1 matches git's own hash of a known file")
    func gitBlobSHA1() throws {
        // `printf 'hello\n' | git hash-object --stdin` → ce013625030ba8dba906f756967f9e9ca394464a
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try Data("hello\n".utf8).write(to: url)
        try Integrity.verify(file: url, displayPath: "hello", size: 6,
                             expected: .gitBlobSHA1("ce013625030ba8dba906f756967f9e9ca394464a"))
        #expect(throws: Integrity.Failure.self) {
            try Integrity.verify(file: url, displayPath: "hello", size: 6,
                                 expected: .gitBlobSHA1(String(repeating: "0", count: 40)))
        }
    }

    @Test("A valid safetensors header lists its tensors")
    func headerReads() throws {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try makeSafetensors([("a", 4), ("b", 8)]).write(to: url)
        let header = try SafetensorsHeader.read(url)
        #expect(header.names == ["a", "b"])
        try header.require(["a"])
        #expect(throws: SafetensorsHeader.Failure.self) { try header.require(["c"]) }
    }

    @Test("A truncated safetensors file is caught from its header alone")
    func truncatedDetected() throws {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let whole = makeSafetensors([("a", 1024)])
        try whole.prefix(whole.count - 100).write(to: url)
        #expect(throws: SafetensorsHeader.Failure.self) { _ = try SafetensorsHeader.read(url) }
    }

    @Test("A byte range that disagrees with the tensor's shape is caught")
    func inconsistentShape() throws {
        let json = #"{"w":{"dtype":"F32","shape":[4],"data_offsets":[0,8]}}"#
        var data = Data()
        var length = UInt64(json.utf8.count).littleEndian
        withUnsafeBytes(of: &length) { data.append(contentsOf: $0) }
        data.append(Data(json.utf8))
        data.append(Data(repeating: 0, count: 8))
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try data.write(to: url)
        #expect(throws: SafetensorsHeader.Failure.inconsistentTensor("w")) {
            _ = try SafetensorsHeader.read(url)
        }
    }

    @Test("The snapshot parser reads LFS SHA-256s and git blob ids from the real API shape")
    func parsesRealSnapshot() throws {
        // Trimmed from GET huggingface.co/api/models/mlx-community/SeedVR2-3B-mlx-int8?blobs=true
        let json = """
        {"sha":"9dbcb304916d94b4a9769b588ea28439054ce3bb","gated":false,"siblings":[
          {"rfilename":"config.json","blobId":"523b29684b54f6807ffc8634bbb3a4d8d923876b","size":247},
          {"rfilename":"vae.safetensors","blobId":"c068ae62a112a45102c6f69ba8b54dca8234efd3","size":501324772,
           "lfs":{"sha256":"a2efbfb38a82a99a9cd16432ca5d564510678d228ce6209692d528aa32285cfb","size":501324772,"pointerSize":134}}
        ]}
        """
        let snapshot = try HuggingFaceHub.parseSnapshot(Data(json.utf8), repo: seedRepo)
        #expect(snapshot.revision == "9dbcb304916d94b4a9769b588ea28439054ce3bb")
        #expect(!snapshot.gated)
        let vae = snapshot.files.first { $0.path == "vae.safetensors" }
        #expect(vae?.expected == .sha256("a2efbfb38a82a99a9cd16432ca5d564510678d228ce6209692d528aa32285cfb"))
        let config = snapshot.files.first { $0.path == "config.json" }
        #expect(config?.expected == .gitBlobSHA1("523b29684b54f6807ffc8634bbb3a4d8d923876b"))
    }

    @Test("Manifest globs select components out of a large multi-pipeline repo")
    func manifestGlobs() {
        // H3's repo ships three pipelines; one mode needs one of them.
        let paths = ["README.md", "model_index.json",
                     "transformer/diffusion_pytorch_model-00001-of-00014.safetensors",
                     "transformer_ref/diffusion_pytorch_model-00001-of-00014.safetensors",
                     "FL2VA/transformer/model-00001-of-00013.safetensors",
                     "vae/diffusion_pytorch_model-00001-of-00003.safetensors"]
        let manifest = ComponentManifest(include: ["model_index.json", "transformer/*", "vae/*"])
        #expect(Set(manifest.select(from: paths)) == [
            "model_index.json",
            "transformer/diffusion_pytorch_model-00001-of-00014.safetensors",
            "vae/diffusion_pytorch_model-00001-of-00003.safetensors",
        ])
    }
}
