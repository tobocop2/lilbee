#!/usr/bin/env python3
"""Grade an answer file against ground_truth.json. Usage: grade.py q1|q2 answer.txt"""
import json, re, sys
gt = json.load(open(__file__.rsplit('/',1)[0] + '/ground_truth.json'))[sys.argv[1]]
ans = open(sys.argv[2]).read()
score, out = 0, []
def hit(pats): return any(re.search(p, ans, re.I) for p in pats)
for f in gt["must_contain_facts"]:
    ok = hit(f["patterns"]); score += 2*ok
    out.append(f"{'PASS' if ok else 'FAIL'} [must]   {f['id']}: {f['desc']}")
for f in gt["should_contain_facts"]:
    ok = hit(f["patterns"]); score += ok
    out.append(f"{'PASS' if ok else 'miss'} [should] {f['id']}: {f['desc']}")
for f in gt["must_not"]:
    bad = hit(f["patterns"]); score -= 3*bad
    out.append(f"{'FAIL' if bad else 'PASS'} [never]  {f['id']}: {f['desc']}")
maxs = 2*len(gt["must_contain_facts"]) + len(gt["should_contain_facts"])
print("\n".join(out)); print(f"SCORE {score}/{maxs}")
sys.exit(0 if (score >= maxs - 2 and all('FAIL' not in l for l in out)) else 1)
