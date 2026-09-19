import LocalLabCore
import SwiftUI

/// Settings → Chat: how long an unused model stays in memory.
struct ChatSettingsView: View {
    @Bindable var model: ChatModel

    var body: some View {
        Form {
            Section {
                Picker("Unload an idle model after", selection: $model.idleUnloadMinutes) {
                    ForEach(ChatModel.idleUnloadChoices, id: \.self) { minutes in
                        Text(label(minutes)).tag(minutes)
                    }
                }
                Text(explanation)
                    .font(.callout).foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            } header: {
                Text("Memory")
            }
            Section {
                HStack {
                    if let repo = model.loadedRepo {
                        Text("\(name(repo)) is loaded, holding \(Format.bytes(model.residentBytes)).")
                        Spacer()
                        Button("Eject now") { Task { await model.eject() } }
                            .disabled(model.isGenerating)
                    } else {
                        Text("No chat model is loaded.").foregroundStyle(.secondary)
                    }
                }
            }
        }
        .formStyle(.grouped)
        .padding(8)
    }

    private func label(_ minutes: Int) -> String {
        switch minutes {
        case 0: "Never"
        case 1: "1 minute"
        case 60: "1 hour"
        default: "\(minutes) minutes"
        }
    }

    private var explanation: String {
        let common = "Unloading gives the model's memory back to other apps. Your conversation is kept; the next message reloads the model (a few seconds) and reads the conversation once more, since the model's cached reading of it goes with it."
        if model.idleUnloadMinutes == 0 {
            return "The model stays loaded until you eject it or a job needs the Mac — replies start immediately, but the memory stays in use. " + common
        }
        return common + " A job starting always unloads it first."
    }

    private func name(_ repo: String) -> String {
        model.installedSpecs.first { $0.repo == repo }?.displayName ?? String(repo.split(separator: "/").last ?? "")
    }
}
