"""Checks that every relative link and image path in the Markdown files resolves.

    python scripts/check_links.py
"""
import os
import re
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PAT = re.compile(r"\]\(([^)\s]+)\)|src=\"([^\"]+)\"")
bad = 0
for dp, dn, fn in os.walk(ROOT):
    dn[:] = [d for d in dn if d not in (".git", "node_modules", "dist")]
    for f in fn:
        if not f.endswith(".md"):
            continue
        p = os.path.join(dp, f)
        with open(p, encoding="utf8") as fh:
            text = fh.read()
        for m in PAT.finditer(text):
            link = m.group(1) or m.group(2)
            if re.match(r"^(https?:|mailto:|#)", link):
                continue
            target = os.path.normpath(os.path.join(dp, link.split("#")[0]))
            if not os.path.exists(target):
                print(f"{os.path.relpath(p, ROOT)}: broken link {link}")
                bad += 1
print("broken links:", bad)
sys.exit(1 if bad else 0)
