import LocalLabCore
import SwiftUI

/// *This Mac*: the facts, what they mean for each task, and the measurement behind the
/// times. Every verdict shows its arithmetic — the dot is not the feature (plan §5.2).
public struct MachineView: View {
    let model: MachineModel

    public init(model: MachineModel) {
        self.model = model
    }

    public var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                header
                capabilities
                measurement
            }
            .padding(20)
            .frame(maxWidth: 760, alignment: .leading)
            .frame(maxWidth: .infinity)
        }
        .navigationTitle("This Mac")
    }

    // MARK: - Facts

    private var header: some View {
        let hardware = model.profile.hardware
        let power = model.profile.power
        return GroupBox {
            VStack(alignment: .leading, spacing: 12) {
                HStack(alignment: .firstTextBaseline, spacing: 10) {
                    Image(systemName: power.hasBattery ? "laptopcomputer" : "desktopcomputer")
                        .font(.title2).foregroundStyle(.secondary)
                    VStack(alignment: .leading, spacing: 2) {
                        Text(hardware.chipName).font(.title2.weight(.semibold))
                        Text(subtitle(hardware)).foregroundStyle(.secondary)
                    }
                }
                Grid(alignment: .leading, horizontalSpacing: 24, verticalSpacing: 6) {
                    fact("Memory bandwidth", String(format: "%.0f GB/s", hardware.memoryBandwidthGBps),
                         note: "spec sheet")
                    fact("The GPU can use", String(format: "%.1f GB", hardware.usableMemoryGB),
                         note: "the lower of its wired limit and memory less 2 GB for macOS")
                    fact("Free right now", String(format: "%.1f GB", hardware.availableMemoryGB),
                         note: "what plans are made against")
                    if let free = model.libraryFreeBytes {
                        fact("Library disk free", Format.bytes(free), note: nil)
                    }
                    fact("Power", powerText(power), note: nil)
                    fact("macOS", hardware.macOSVersionString, note: nil)
                }
                .font(.callout)
                Text(model.profile.timing.summary).font(.caption).foregroundStyle(.secondary)
            }
            .padding(6)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
    }

    private func subtitle(_ hardware: HardwareProfile) -> String {
        var parts: [String] = []
        if let cores = hardware.gpuCoreCount { parts.append("\(cores)-core GPU") }
        parts.append(String(format: "%.0f GB memory", hardware.totalMemoryGB))
        return parts.joined(separator: " · ")
    }

    private func powerText(_ power: PowerState) -> String {
        guard let percent = power.batteryPercent else { return "Mains" }
        return power.onBattery ? "Battery, \(percent)%" : "Power adapter, battery \(percent)%"
    }

    @ViewBuilder
    private func fact(_ label: String, _ value: String, note: String?) -> some View {
        GridRow {
            Text(label).foregroundStyle(.secondary)
            Text(value).monospacedDigit()
            Text(note ?? "").font(.caption).foregroundStyle(.tertiary)
        }
    }

    // MARK: - Capabilities

    private var capabilities: some View {
        GroupBox {
            VStack(alignment: .leading, spacing: 14) {
                ForEach(model.profile.capabilities) { capability in
                    CapabilityRow(capability: capability)
                    if capability.id != model.profile.capabilities.last?.id { Divider() }
                }
            }
            .padding(6)
            .frame(maxWidth: .infinity, alignment: .leading)
        } label: {
            Label("What this Mac can do", systemImage: "checklist")
        }
    }

    // MARK: - Measurement

    private var measurement: some View {
        GroupBox {
            VStack(alignment: .leading, spacing: 12) {
                switch model.probeState {
                case .running(let completed, let total, let current):
                    HStack(spacing: 12) {
                        ProgressView(value: Double(completed), total: Double(max(total, 1)))
                            .frame(maxWidth: 260)
                        Text(current.map { "Measuring \($0.title.lowercased())…" } ?? "Finishing…")
                            .foregroundStyle(.secondary)
                        Spacer()
                        Button("Stop") { model.stopMeasuring() }
                    }
                    Text("A few seconds of small GPU tests — no download, nothing leaves this Mac.")
                        .font(.caption).foregroundStyle(.secondary)
                case .failed(let message):
                    Label(message, systemImage: "exclamationmark.triangle").foregroundStyle(.orange)
                    measureButton(title: "Try again")
                case .idle:
                    if let report = model.probes {
                        ProbeTable(report: report, reference: ReferenceMachine.all.first?.probes)
                        HStack {
                            Text("Measured \(report.date.formatted(date: .abbreviated, time: .shortened)) in \(Format.duration(report.durationSeconds)) · macOS \(report.macOSVersion)")
                                .font(.caption).foregroundStyle(.secondary)
                            Spacer()
                            measureButton(title: "Measure again")
                        }
                    } else {
                        Text("LocalLab hasn't measured this Mac yet. Until it does, times are scaled from another Mac by spec sheet, which is only a rough guide.")
                            .fixedSize(horizontal: false, vertical: true)
                        if let reason = model.autoMeasureBlocked {
                            Text(reason).font(.caption).foregroundStyle(.secondary)
                        }
                        measureButton(title: "Measure this Mac")
                    }
                }
            }
            .padding(6)
            .frame(maxWidth: .infinity, alignment: .leading)
        } label: {
            Label("Measured speed", systemImage: "speedometer")
        }
    }

    private func measureButton(title: String) -> some View {
        Button(title) { model.measure() }
            .disabled(model.isMeasuring)
    }
}

private struct CapabilityRow: View {
    let capability: MachineProfile.Capability

    var body: some View {
        HStack(alignment: .firstTextBaseline, spacing: 12) {
            Image(systemName: symbol).foregroundStyle(color).font(.title3)
                .accessibilityLabel(statusText)
            VStack(alignment: .leading, spacing: 3) {
                HStack(alignment: .firstTextBaseline) {
                    Text(capability.title).font(.headline)
                    Text(capability.example).font(.caption).foregroundStyle(.secondary)
                    Spacer()
                    Text(statusText).font(.caption2)
                        .padding(.horizontal, 6).padding(.vertical, 1)
                        .background(Capsule().fill(color.opacity(0.15)))
                }
                Text(capability.headline)
                Text(capability.detail).font(.caption).foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
    }

    private var symbol: String {
        switch capability.status {
        case .ready: "checkmark.circle.fill"
        case .closeApps: "exclamationmark.circle.fill"
        case .tooLarge: "xmark.circle.fill"
        case .notBuilt: "clock"
        }
    }

    private var color: Color {
        switch capability.status {
        case .ready: .green
        case .closeApps: .yellow
        case .tooLarge: .red
        case .notBuilt: .gray
        }
    }

    private var statusText: String {
        switch capability.status {
        case .ready: "Runs now"
        case .closeApps: "Close other apps"
        case .tooLarge: "Too large"
        case .notBuilt: "Coming"
        }
    }
}

/// The probe results, each against the reference Mac's, so another Mac can see at a glance
/// how it compares with the one the times came from.
private struct ProbeTable: View {
    let report: ProbeReport
    let reference: ProbeReport?

    var body: some View {
        Grid(alignment: .leading, horizontalSpacing: 16, verticalSpacing: 6) {
            GridRow {
                Text("Test").foregroundStyle(.secondary)
                Text("Result").foregroundStyle(.secondary).gridColumnAlignment(.trailing)
                Text(referenceHeading).foregroundStyle(.secondary).gridColumnAlignment(.trailing)
                Text("Stands in for").foregroundStyle(.secondary)
            }
            .font(.caption)
            ForEach(report.results, id: \.kind) { result in
                GridRow {
                    Text(result.kind.title)
                    Text(String(format: "%.1f %@", result.value, result.kind.unit)).monospacedDigit()
                    Text(ratio(result)).monospacedDigit().foregroundStyle(.secondary)
                    Text(result.kind.standsInFor).font(.caption).foregroundStyle(.secondary)
                        .lineLimit(2).help(result.detail)
                }
            }
        }
        .font(.callout)
    }

    private var referenceHeading: String {
        guard let reference else { return "" }
        return "vs " + reference.machine.chipName.replacingOccurrences(of: "Apple ", with: "")
    }

    private func ratio(_ result: ProbeResult) -> String {
        guard result.kind.isThroughput, let theirs = reference?.value(result.kind), theirs > 0 else { return "" }
        return String(format: "%.2f×", result.value / theirs)
    }
}
