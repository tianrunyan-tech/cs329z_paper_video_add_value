"""Small controlled experiment inspired by ACE (Zhang et al., ICLR 2026).

Question: does removing ground-truth feedback from the Reflector change the
quality of the lessons ACE learns, and accuracy on unseen questions?

Stages (run in order; every model call is cached, so reruns are free):
  python3 ace_exp.py pilot     # 21 train questions, baseline sanity check (not used later)
  python3 ace_exp.py traj      # Generator answers the 21 build questions once, no playbook
  python3 ace_exp.py build     # Reflector + Curator build playbook B (labels) and C (no labels)
  python3 ace_exp.py test      # conditions A, B, C on the same 49 test questions
  python3 ace_exp.py paired    # per-question changes vs. baseline + exact McNemar test
  python3 ace_exp.py ablate    # post-hoc: remove suspect bullets, retest
  python3 ace_exp.py audit     # export all bullets, shuffled and anonymised, for blind grading
  python3 ace_exp.py reveal out/audit_blind.csv   # after grading: join grades with condition and count
"""
import csv, hashlib, json, os, random, re, sys, time
from pathlib import Path

from openai import OpenAI

ROOT = Path(__file__).parent
DATA = ROOT / "data"
OUT = ROOT / "out"
OUT.mkdir(exist_ok=True)

MODEL = os.environ.get("ACE_MODEL", "openai/gpt-4o-mini")
SEED = 329
LABELS = [
    "pending_transfer",
    "transfer_not_received_by_recipient",
    "transfer_timing",
    "balance_not_updated_after_bank_transfer",
    "failed_transfer",
    "declined_transfer",
    "cancel_transfer",
]
N_PILOT, N_BUILD, N_TEST = 3, 3, 7  # per class
MAX_NEW_BULLETS = 3

client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ.get("OPENROUTER_API_KEY", "dummy-key"))
CACHE = OUT / f"cache_{MODEL.replace('/', '_')}.jsonl"
_cache = {}
if CACHE.exists():
    for line in CACHE.open():
        rec = json.loads(line)
        _cache[rec["key"]] = rec
STATS = {"calls": 0, "cached": 0, "prompt_tokens": 0, "completion_tokens": 0}


def call(system, user, max_tokens=600):
    key = hashlib.sha256(json.dumps([MODEL, system, user, max_tokens]).encode()).hexdigest()
    if key in _cache:
        STATS["cached"] += 1
        return _cache[key]["text"]
    for attempt in range(5):
        try:
            r = client.chat.completions.create(
                model=MODEL,
                temperature=0,
                max_tokens=max_tokens,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            )
            text = r.choices[0].message.content or ""
            usage = r.usage
            break
        except Exception as e:  # network or rate-limit errors: back off and retry
            print(f"  retry {attempt + 1}: {e}", file=sys.stderr)
            time.sleep(2 ** attempt)
    else:
        raise RuntimeError("model call failed 5 times")
    rec = {
        "key": key,
        "text": text,
        "prompt_tokens": getattr(usage, "prompt_tokens", 0),
        "completion_tokens": getattr(usage, "completion_tokens", 0),
    }
    _cache[key] = rec
    with CACHE.open("a") as f:
        f.write(json.dumps(rec) + "\n")
    STATS["calls"] += 1
    STATS["prompt_tokens"] += rec["prompt_tokens"]
    STATS["completion_tokens"] += rec["completion_tokens"]
    return text


def parse_json(text):
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------- data split
def load_split():
    """Stratified, seeded split. Pilot and build come from train, test from test."""
    rng = random.Random(SEED)

    def by_label(path):
        groups = {l: [] for l in LABELS}
        seen = set()
        for row in csv.DictReader(path.open()):
            t = row["text"].strip()
            if row["category"] in groups and t.lower() not in seen:
                seen.add(t.lower())
                groups[row["category"]].append(t)
        return groups

    train, test = by_label(DATA / "train.csv"), by_label(DATA / "test.csv")
    test_texts = {t.lower() for g in test.values() for t in g}
    pilot, build, held = [], [], []
    for l in LABELS:
        tr = [t for t in train[l] if t.lower() not in test_texts]
        rng.shuffle(tr)
        pilot += [{"text": t, "label": l} for t in tr[:N_PILOT]]
        build += [{"text": t, "label": l} for t in tr[N_PILOT:N_PILOT + N_BUILD]]
        te = list(test[l])
        rng.shuffle(te)
        held += [{"text": t, "label": l} for t in te[:N_TEST]]
    for name, items in [("pilot", pilot), ("build", build), ("test", held)]:
        rng.shuffle(items)
        for i, it in enumerate(items):
            it["id"] = f"{name}-{i:02d}"
    return pilot, build, held


# ---------------------------------------------------------------- roles
LABEL_LIST = "\n".join(f"- {l}" for l in LABELS)


def generator(query, playbook):
    pb = ""
    if playbook:
        pb = "\n\nPlaybook of lessons learned from earlier queries (use them when relevant):\n" + "\n".join(
            f"[{b['id']}] {b['text']}" for b in playbook
        )
    system = (
        "You classify customer messages to a bank into exactly one intent label.\n"
        f"Allowed labels:\n{LABEL_LIST}{pb}\n\n"
        'Reply with JSON only: {"reasoning": "<one or two sentences>", "label": "<one allowed label>"}'
    )
    out = parse_json(call(system, f"Customer message: {query}", max_tokens=300)) or {}
    label = out.get("label", "").strip()
    return {"reasoning": out.get("reasoning", ""), "label": label if label in LABELS else "INVALID"}


def reflector(query, attempt, gold):
    """The only intended difference between B and C is the `gold` argument."""
    system = (
        "You review a classification attempt by an assistant that sorts bank customer messages "
        "into intent labels, and extract lessons that would help it classify similar messages "
        f"correctly in future.\nAllowed labels:\n{LABEL_LIST}\n\n"
        "Lessons must be specific and actionable: name the labels involved and the cue that "
        "separates them. Write at most 3 lessons.\n"
        'Reply with JSON only: {"analysis": "<what the attempt got right or wrong and why>", '
        '"lessons": ["<lesson>", ...]}'
    )
    user = (
        f"Customer message: {query}\n"
        f"Assistant's reasoning: {attempt['reasoning']}\n"
        f"Assistant's label: {attempt['label']}\n"
    )
    if gold is not None:
        user += f"Correct label: {gold}\n"
    else:
        user += "The correct label is not available. Judge the attempt on its own merits.\n"
    out = parse_json(call(system, user, max_tokens=500)) or {}
    return {"analysis": out.get("analysis", ""), "lessons": [str(x) for x in out.get("lessons", [])][:3]}


def curator(playbook, lessons):
    system = (
        "You maintain a playbook of short, specific lessons for an assistant that classifies bank "
        "customer messages into intent labels. You receive the current playbook and new candidate "
        "lessons. Output only the lessons worth ADDING as new bullets: skip any that repeat an "
        f"existing bullet. Do not rewrite existing bullets. Add at most {MAX_NEW_BULLETS}.\n"
        'Reply with JSON only: {"new_bullets": ["<bullet>", ...]}'
    )
    current = "\n".join(f"[{b['id']}] {b['text']}" for b in playbook) or "(empty)"
    cand = "\n".join(f"- {l}" for l in lessons) or "(none)"
    out = parse_json(call(system, f"Current playbook:\n{current}\n\nCandidate lessons:\n{cand}", max_tokens=400)) or {}
    return [str(x) for x in out.get("new_bullets", [])][:MAX_NEW_BULLETS]


# ---------------------------------------------------------------- stages
def save(name, obj):
    (OUT / name).write_text(json.dumps(obj, indent=2, ensure_ascii=False))


def load(name):
    return json.loads((OUT / name).read_text())


def stage_pilot():
    pilot, _, _ = load_split()
    rows = []
    for it in pilot:
        a = generator(it["text"], [])
        rows.append({**it, "pred": a["label"], "reasoning": a["reasoning"], "correct": a["label"] == it["label"]})
        print(f"{'OK ' if rows[-1]['correct'] else 'X  '} {it['label']:<40} -> {a['label']}")
    acc = sum(r["correct"] for r in rows) / len(rows)
    print(f"\nPilot baseline: {sum(r['correct'] for r in rows)}/{len(rows)} = {acc:.0%}  (model {MODEL})")
    save("pilot.json", rows)


def stage_traj():
    _, build, _ = load_split()
    traj = []
    for it in build:
        a = generator(it["text"], [])
        traj.append({**it, "attempt": a, "correct": a["label"] == it["label"]})
    save("trajectories.json", traj)
    print(f"Build-set answers (no playbook): {sum(t['correct'] for t in traj)}/{len(traj)} correct")


def stage_build():
    traj = load("trajectories.json")
    for cond, use_gold in [("B", True), ("C", False)]:
        playbook, log = [], []
        for t in traj:
            r = reflector(t["text"], t["attempt"], t["label"] if use_gold else None)
            new = curator(playbook, r["lessons"])
            added = []
            for text in new:
                b = {
                    "id": f"{cond}{len(playbook) + 1:02d}",
                    "text": text,
                    "source": t["id"],
                    "source_correct": t["correct"],
                }
                playbook.append(b)
                added.append(b["id"])
            log.append({"source": t["id"], "reflection": r, "added": added})
        save(f"playbook_{cond}.json", playbook)
        save(f"build_log_{cond}.json", log)
        print(f"Playbook {cond}: {len(playbook)} bullets")


def stage_test():
    _, _, test = load_split()
    books = {"A": [], "B": load("playbook_B.json"), "C": load("playbook_C.json")}
    results = {}
    for cond, pb in books.items():
        rows = []
        for it in test:
            a = generator(it["text"], pb)
            rows.append({**it, "pred": a["label"], "reasoning": a["reasoning"], "correct": a["label"] == it["label"]})
        results[cond] = rows
        n = sum(r["correct"] for r in rows)
        print(f"Condition {cond}: {n}/{len(rows)} = {n / len(rows):.1%}  (playbook bullets: {len(pb)})")
    save("test_results.json", results)
    # per-class accuracy, to see where differences come from
    print("\nPer-class correct (A / B / C), out of", N_TEST)
    for l in LABELS:
        c = [sum(r["correct"] for r in results[k] if r["label"] == l) for k in "ABC"]
        print(f"  {l:<40} {c[0]} / {c[1]} / {c[2]}")


def mcnemar_exact(b, c):
    """Two-sided exact McNemar p-value from the discordant counts b and c."""
    from math import comb
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def stage_paired():
    """Per-question transitions against the baseline (no new model calls)."""
    r = load("test_results.json")
    out = {}
    for cond in "BC":
        t = {"A right -> wrong": 0, "A wrong -> right": 0, "both right": 0, "both wrong": 0}
        for a, x in zip(r["A"], r[cond]):
            assert a["id"] == x["id"]
            key = {(1, 0): "A right -> wrong", (0, 1): "A wrong -> right", (1, 1): "both right", (0, 0): "both wrong"}[
                (int(a["correct"]), int(x["correct"]))
            ]
            t[key] += 1
        t["mcnemar_p"] = round(mcnemar_exact(t["A right -> wrong"], t["A wrong -> right"]), 3)
        out[cond] = t
        print(cond, t)
    save("paired.json", out)


# Post-hoc diagnostic: these bullets were chosen AFTER looking at the test results.
ABLATIONS = {"C": ["C15", "C21"], "B": ["B24"]}


def stage_ablate():
    _, _, test = load_split()
    r = load("test_results.json")
    res = {}
    for cond, drop in ABLATIONS.items():
        pb = [b for b in load(f"playbook_{cond}.json") if b["id"] not in drop]
        rows = []
        for it in test:
            a = generator(it["text"], pb)
            rows.append({**it, "pred": a["label"], "reasoning": a["reasoning"], "correct": a["label"] == it["label"]})
        res[cond] = {"dropped": drop, "rows": rows}
        n = sum(x["correct"] for x in rows)
        print(f"{cond} without {drop}: {n}/{len(rows)}  (full playbook: {sum(x['correct'] for x in r[cond])}/{len(rows)}, baseline: {sum(x['correct'] for x in r['A'])})")
        for full, abl in zip(r[cond], rows):
            if full["pred"] != abl["pred"]:
                print(f"   gold={full['label']:<38} full={full['pred']:<38} ablated={abl['pred']}  | {full['text']}")
    save("ablation_results.json", res)


def stage_audit():
    rng = random.Random(SEED + 1)
    bullets = load("playbook_B.json") + load("playbook_C.json")
    rng.shuffle(bullets)
    key = []
    with (OUT / "audit_blind.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["item", "bullet", "grade (sound / harmful / vague)", "reason or counterexample"])
        for i, b in enumerate(bullets):
            w.writerow([f"L{i + 1:02d}", b["text"], "", ""])
            key.append({"item": f"L{i + 1:02d}", **b})
    save("audit_key.json", key)
    print(f"Wrote {len(bullets)} anonymised bullets to out/audit_blind.csv. Grade them before running 'reveal'.")


def stage_reveal(graded_path):
    key = {k["item"]: k for k in load("audit_key.json")}
    rows = list(csv.DictReader(open(graded_path)))
    table = {}
    for r in rows:
        k = key[r["item"]]
        cond = k["id"][0]
        g = r["grade (sound / harmful / vague)"].strip().lower()
        t = table.setdefault(cond, {"total": 0, "sound": 0, "harmful": 0, "vague": 0, "from_wrong": 0, "harmful_from_wrong": 0})
        t["total"] += 1
        t[g] = t.get(g, 0) + 1
        if not k["source_correct"]:
            t["from_wrong"] += 1
            if g == "harmful":
                t["harmful_from_wrong"] += 1
    save("audit_summary.json", table)
    print(json.dumps(table, indent=2))


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else ""
    fn = {
        "pilot": stage_pilot,
        "traj": stage_traj,
        "build": stage_build,
        "test": stage_test,
        "paired": stage_paired,
        "ablate": stage_ablate,
        "audit": stage_audit,
    }.get(stage)
    if stage == "reveal":
        stage_reveal(sys.argv[2])
    elif fn:
        fn()
    else:
        print(__doc__)
        sys.exit(1)
    print(f"[model calls: {STATS['calls']} new, {STATS['cached']} cached; "
          f"tokens in/out: {STATS['prompt_tokens']}/{STATS['completion_tokens']}]")
