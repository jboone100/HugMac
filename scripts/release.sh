#!/bin/zsh
# Build a release of LocalLab for the Mac App Store.
#
#   scripts/release.sh            archive, check, and export a signed package to build/export
#   scripts/release.sh --upload   the same, then upload it to App Store Connect
#
# Uploading is never the default: it sends the build to Apple. It needs Xcode signed in to the
# developer account (Settings → Accounts) so an Apple Distribution certificate can be used or
# created. An uploaded build appears in App Store Connect and TestFlight; it goes to review
# only when submitted there.
set -euo pipefail
cd "$(dirname "$0")/.."

upload=false
[[ "${1:-}" == "--upload" ]] && upload=true

# Every upload needs a higher build number than the last: the commit count only grows.
build_number=$(git rev-list --count HEAD)
version=$(grep MARKETING_VERSION project.yml | head -1 | sed -E 's/.*"(.*)".*/\1/')
if [[ -n "$(git status --porcelain)" ]]; then
  echo "note: uncommitted changes — the build number ($build_number) won't identify this build exactly" >&2
fi

archive=build/LocalLab.xcarchive
rm -rf build/LocalLab.xcarchive build/export
mkdir -p build

echo "── LocalLab $version ($build_number): archiving the Release build"
xcodegen generate >/dev/null
xcodebuild archive \
  -project LocalLab.xcodeproj -scheme LocalLab -configuration Release \
  -archivePath "$archive" -destination 'generic/platform=macOS' \
  -allowProvisioningUpdates \
  CURRENT_PROJECT_VERSION="$build_number" \
  | grep -E "error:|warning: .*(sign|entitle)|ARCHIVE (SUCCEEDED|FAILED)" || true
[[ -d "$archive" ]] || { echo "archive failed"; exit 1; }

app="$archive/Products/Applications/LocalLab.app"
echo "── checking $app"
# Every check stops the release with a sentence. Outputs are captured before matching:
# piping into `grep -q` under pipefail fails whenever grep stops reading early.
fail() { echo "  ✗ $1"; exit 1; }
codesign --verify --deep --strict "$app" 2>/dev/null || fail "signature doesn't verify"
echo "  ✓ signature valid"
signature=$(codesign -dv "$app" 2>&1)
[[ "$signature" == *"TeamIdentifier=K8CKJEN3KC"* ]] || fail "not signed by team K8CKJEN3KC"
[[ "$signature" == *"(runtime)"* ]] || fail "hardened runtime is off"
echo "  ✓ team K8CKJEN3KC, hardened runtime"
entitlements=$(codesign -d --entitlements - "$app" 2>/dev/null)
for key in app-sandbox network.client files.user-selected.read-write files.bookmarks.app-scope; do
  [[ "$entitlements" == *"com.apple.security.$key"* ]] || fail "missing entitlement $key"
done
[[ "$entitlements" != *"get-task-allow"* ]] || fail "debug entitlement present — not a release build"
echo "  ✓ entitlements: sandbox, network client, user-selected files, bookmarks"
[[ "$(lipo -archs "$app/Contents/MacOS/LocalLab")" == "arm64" ]] || fail "not arm64-only"
echo "  ✓ arm64 only"
[[ -f "$app/Contents/Resources/PrivacyInfo.xcprivacy" ]] || fail "privacy manifest missing"
[[ -f "$app/Contents/Resources/AppIcon.icns" ]] || fail "icon missing"
[[ -n "$(find "$app" -name default.metallib)" ]] || fail "MLX kernels (default.metallib) missing"
echo "  ✓ privacy manifest, icon, MLX kernels"
binary_strings=$(strings "$app/Contents/MacOS/LocalLab")
[[ "$binary_strings" != *"LOCALLAB_SANDBOX_CHECK"* ]] || fail "debug hooks compiled in — not a release build"
echo "  ✓ no debug hooks"
info="$app/Contents/Info.plist"
echo "  ✓ version $(/usr/libexec/PlistBuddy -c 'Print :CFBundleShortVersionString' "$info") ($(/usr/libexec/PlistBuddy -c 'Print :CFBundleVersion' "$info")), minimum macOS $(/usr/libexec/PlistBuddy -c 'Print :LSMinimumSystemVersion' "$info")"

# Not "options": zsh reserves that name for its own settings.
export_options=scripts/export/app-store.plist
if $upload; then
  export_options=build/app-store-upload.plist
  sed 's#<string>export</string>#<string>upload</string>#' scripts/export/app-store.plist > "$export_options"
  echo "── uploading to App Store Connect"
else
  echo "── exporting a signed App Store package"
fi
xcodebuild -exportArchive -archivePath "$archive" -exportPath build/export \
  -exportOptionsPlist "$export_options" -allowProvisioningUpdates > build/export.log 2>&1 || true
grep -E "error:|EXPORT (SUCCEEDED|FAILED)|Upload succeeded|Uploaded" build/export.log | sed 's/^/  /' || true
ls build/export 2>/dev/null | sed 's/^/  /'
