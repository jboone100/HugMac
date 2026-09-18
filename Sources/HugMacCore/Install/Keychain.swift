import Foundation
import Security

/// The Hugging Face token, in the macOS Keychain — never in a file, never logged.
///
/// HugMac's own service name, so it neither reads nor clobbers MLXUI's `com.ai-browser` item.
public enum Keychain {
    static let service = "com.hugmac"
    static let huggingFaceAccount = "huggingface-token"

    public static func huggingFaceToken() -> String? { get(account: huggingFaceAccount) }
    public static func saveHuggingFaceToken(_ token: String) { save(token, account: huggingFaceAccount) }
    public static func deleteHuggingFaceToken() { delete(account: huggingFaceAccount) }

    static func get(account: String) -> String? {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: account,
            kSecReturnData as String: true,
            kSecMatchLimit as String: kSecMatchLimitOne,
        ]
        var item: CFTypeRef?
        guard SecItemCopyMatching(query as CFDictionary, &item) == errSecSuccess,
              let data = item as? Data else { return nil }
        return String(data: data, encoding: .utf8)
    }

    static func save(_ value: String, account: String) {
        guard let data = value.data(using: .utf8) else { return }
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: account,
        ]
        SecItemDelete(query as CFDictionary)
        var add = query
        add[kSecValueData as String] = data
        SecItemAdd(add as CFDictionary, nil)
    }

    static func delete(account: String) {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: account,
        ]
        SecItemDelete(query as CFDictionary)
    }
}
