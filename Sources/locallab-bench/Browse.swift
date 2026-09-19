import Foundation
import LocalLabCore

/// `locallab-bench --browse [search] [--task chat|vision|…] [--all] [--open repo]` — the Browse
/// screen's list against the live Hugging Face API, graded for this Mac.
enum BrowseCommand {
    static func run(arguments: [String]) async throws {
        var query = CatalogQuery()
        var open: String?
        var rest = arguments[...]
        while let argument = rest.popFirst() {
            switch argument {
            case "--task": query.task = rest.popFirst().flatMap(CatalogTask.init(rawValue:))
            case "--all": query.publisher = .everyone
            case "--open": open = rest.popFirst()
            default: query.text = argument
            }
        }
        let hardware = HardwareProfile.detect()
        let calibration = CalibrationStore.load()
        let catalog = HuggingFaceCatalog()
        if let open {
            let entry = try await catalog.search(CatalogQuery(text: String(open.split(separator: "/").last ?? ""),
                                                              publisher: .everyone), pageSize: 20)
                .entries.first { $0.repo == open } ?? CatalogEntry(repo: open)
            let details = try await catalog.details(repo: open)
            let before = ModelGrader.verdict(for: entry, hardware: hardware, calibration: calibration)
            let after = ModelGrader.verdict(for: entry, details: details, hardware: hardware, calibration: calibration)
            print(open)
            print("  listing:  \(before.headline) — \(before.arithmetic)")
            print("  opened:   \(after.headline) — \(after.arithmetic)")
            print("  exact weights \(gb(details.weightBytes)), download \(gb(details.downloadBytes)); config: \(details.config.map { "\($0.layers) layers, \($0.cacheLayers) cached, \($0.kvHeads)×\($0.headDim) KV" } ?? "none")")
            return
        }
        let page = try await catalog.search(query, pageSize: 50)
        let rows = page.entries.map { ($0, ModelGrader.verdict(for: $0, hardware: hardware, calibration: calibration)) }
            .sorted(by: ModelGrader.bestFitOrder)
        print("\(rows.count) results for \(hardware.machineKey.displayName) — best fit first\n")
        for (entry, verdict) in rows.prefix(25) {
            let dot: String = switch verdict.grade {
            case .green: "🟢"
            case .yellow: "🟡"
            case .red: "🔴"
            case .notRunnable: "⚪️"
            }
            let size = verdict.weightBytes.map { (verdict.isEstimate ? "~" : "") + gb($0) } ?? "?"
            print("\(dot) \(entry.repo.padding(toLength: 52, withPad: " ", startingAt: 0)) \(size.padding(toLength: 10, withPad: " ", startingAt: 0)) \(verdict.runner.label.padding(toLength: 8, withPad: " ", startingAt: 0)) \(verdict.headline)")
        }
    }
}
