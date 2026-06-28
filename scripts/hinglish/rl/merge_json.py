#!/usr/bin/env python3
"""Merge two score jsons (cached + new) by concatenating their 'rows' -> a powered panel json."""
import json, sys
A = json.load(open(sys.argv[1])); B = json.load(open(sys.argv[2]))
A["rows"] = A["rows"] + B["rows"]
A["n"] = len(A["rows"])
json.dump(A, open(sys.argv[3], "w"), ensure_ascii=False, indent=2)
print(f"merged {len(A['rows'])} rows -> {sys.argv[3]}")
