# tests/fixtures/fake_security.py
"""Stands in for macOS security(1) in tests: add-generic-password and
find-generic-password over the JSON file named by FAKE_KEYCHAIN. With
FAKE_SECURITY_FAIL=1, add-generic-password fails and echoes its arguments
to stderr (so tests prove R3 scrubs secrets from errors)."""

import json
import os
import sys

path = os.environ["FAKE_KEYCHAIN"]
items = json.load(open(path)) if os.path.exists(path) else {}
cmd, rest = sys.argv[1], sys.argv[2:]


def opt(flag):
    return rest[rest.index(flag) + 1] if flag in rest else None


if cmd == "add-generic-password":
    if os.environ.get("FAKE_SECURITY_FAIL"):
        print("security: add failed for " + " ".join(rest), file=sys.stderr)
        sys.exit(1)
    items[opt("-s")] = {"acct": opt("-a"), "pw": opt("-w")}
    json.dump(items, open(path, "w"))
    sys.exit(0)
if cmd == "find-generic-password":
    item = items.get(opt("-s"))
    if item is None:
        print("security: SecKeychainSearchCopyNext: The specified item could not be found in the keychain.",
              file=sys.stderr)
        sys.exit(44)
    if "-w" in rest:
        print(item["pw"])
    else:
        print(f'keychain: "fake"\nattributes:\n    "acct"<blob>="{item["acct"]}"\n    "svce"<blob>="{opt("-s")}"')
    sys.exit(0)
sys.exit(2)
