import AppKit
import HugMacCore
import SwiftUI

/// Every job, newest first: what it's doing, and what can be done to it.
public struct JobsView: View {
    let queue: JobQueue

    public init(queue: JobQueue) {
        self.queue = queue
    }

    public var body: some View {
        Group {
            if queue.jobs.isEmpty {
                ContentUnavailableView(
                    "No jobs yet", systemImage: "tray",
                    description: Text("Upscales you start appear here, and keep running when their window closes.")
                )
            } else {
                TimelineView(.periodic(from: .now, by: 1)) { context in
                    List {
                        ForEach(queue.jobs.reversed()) { job in
                            JobRow(job: job, queue: queue, now: context.date)
                                .padding(.vertical, 6)
                        }
                    }
                }
            }
        }
        .navigationTitle("Jobs")
    }
}

struct JobRow: View {
    let job: Job
    let queue: JobQueue
    let now: Date

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(alignment: .firstTextBaseline) {
                StateBadge(state: job.state)
                Text(job.title).font(.headline).lineLimit(1)
                Spacer()
                actions
            }
            if job.state == .running || job.state == .paused || job.state == .interrupted {
                ProgressView(value: job.progress.fraction)
            }
            Text(detail).font(.callout).foregroundStyle(.secondary)
            if let note = job.note ?? job.failure {
                Text(note).font(.caption)
                    .foregroundStyle(job.state == .failed ? Color.red : Color.secondary)
            }
        }
    }

    @ViewBuilder private var actions: some View {
        HStack(spacing: 6) {
            switch job.state {
            case .running:
                Button("Pause") { queue.pause(job.id) }
                Button("Cancel", role: .cancel) { queue.cancel(job.id) }
            case .queued:
                Button("Hold") { queue.pause(job.id) }
                Button("Cancel", role: .cancel) { queue.cancel(job.id) }
            case .paused, .interrupted, .failed:
                Button("Resume") { queue.resume(job.id) }
                Button("Cancel", role: .cancel) { queue.cancel(job.id) }
            case .completed:
                if let url = job.outcome?.outputURL {
                    Button("Show in Finder") { NSWorkspace.shared.activateFileViewerSelecting([url]) }
                    Button("Open") { NSWorkspace.shared.open(url) }
                }
                Button("Remove") { queue.remove(job.id) }
            case .cancelled:
                Button("Remove") { queue.remove(job.id) }
            }
        }
        .controlSize(.small)
    }

    private var detail: String {
        let elapsed = queue.elapsedSeconds(for: job.id, now: now)
        switch job.state {
        case .queued:
            return "Queued · \(job.createdAt.formatted(date: .omitted, time: .shortened))"
        case .running:
            var parts = [UpscaleModel.phaseLabel(job.progress.phase)]
            if job.progress.unitsTotal > 1 {
                parts.append("chunk \(min(job.progress.unitsDone + 1, job.progress.unitsTotal)) of \(job.progress.unitsTotal)")
            }
            parts.append("\(Format.clock(elapsed)) elapsed")
            if let remaining = queue.remainingSeconds(now: now) {
                parts.append("about \(Format.duration(remaining)) left")
            }
            if job.attempts > 1 { parts.append("resumed") }
            return parts.joined(separator: " · ")
        case .paused, .interrupted:
            return String(format: "%.0f%% done · %@ spent so far · work kept",
                          job.progress.fraction * 100, Format.clock(elapsed))
        case .completed:
            guard let outcome = job.outcome else { return "Done" }
            return "Done in \(Format.duration(elapsed)) · peak \(Format.bytes(outcome.peakBytes))"
                + (job.attempts > 1 ? " · resumed \(job.attempts - 1)×" : "")
        case .failed:
            return "Failed after \(Format.clock(elapsed)) · work kept, so Resume picks up from there"
        case .cancelled:
            return "Cancelled"
        }
    }
}

struct StateBadge: View {
    let state: Job.State

    var body: some View {
        Text(label)
            .font(.caption2.weight(.semibold))
            .padding(.horizontal, 6).padding(.vertical, 2)
            .background(Capsule().fill(color.opacity(0.15)))
            .foregroundStyle(color)
    }

    private var label: String {
        switch state {
        case .queued: "Queued"
        case .running: "Running"
        case .paused: "Paused"
        case .interrupted: "Interrupted"
        case .completed: "Done"
        case .failed: "Failed"
        case .cancelled: "Cancelled"
        }
    }

    private var color: Color {
        switch state {
        case .running: .accentColor
        case .queued: .secondary
        case .paused, .interrupted: .orange
        case .completed: .green
        case .failed: .red
        case .cancelled: .secondary
        }
    }
}
