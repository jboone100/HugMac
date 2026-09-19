#!/usr/bin/env python3
"""Check docs/app-store-listing.md against App Store Connect's character limits."""
import re, sys, pathlib

text = pathlib.Path(__file__).resolve().parent.parent.joinpath("docs/app-store-listing.md").read_text()
limits = {"Name": 30, "Subtitle": 30, "Promotional text": 170, "Description": 4000, "Keywords": 100, "What's New": 4000}
ok = True
for section, limit in limits.items():
    match = re.search(rf"^## {re.escape(section)} \(.*?\)\n\n(.*?)(?=\n## |\Z)", text, re.S | re.M)
    if not match:
        print(f"  ? {section}: not found"); ok = False; continue
    body = match.group(1).strip()
    if section == "Promotional text":
        body = " ".join(body.split("\n"))
    length = len(body)
    flag = "✓" if length <= limit else "✗"
    ok &= length <= limit
    print(f"  {flag} {section}: {length}/{limit}")
    if section == "Keywords":
        words = body.split(",")
        dupes = {w for w in words if words.count(w) > 1}
        if " ," in body or ", " in body: print("    spaces around commas waste characters"); ok = False
        if dupes: print(f"    repeated: {dupes}"); ok = False
sys.exit(0 if ok else 1)
