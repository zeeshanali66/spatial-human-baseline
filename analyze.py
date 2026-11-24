"""Analyse the human survey and compare people with the models on the same items (preregistered in PREREGISTRATION.md).

Input: the Google Forms CSV export. Item columns are found by the [ID] at the start of each question title.
  python analyze.py responses.csv          -> results/RESULTS_HUMAN.md
  python analyze.py --simulate 40          -> sim/ (fake answers, only to test this script)
"""
import json, os, re, sys
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
KEY = {k["id"]: k for k in json.load(open(os.path.join(HERE, "answer_key.json")))}
MOD = json.load(open(os.path.join(HERE, "model_answers.json")))
RNG = np.random.default_rng(0)
B = 5000


def boot(x):
    x = np.asarray(x, float)
    if len(x) == 0:
        return "-"
    bs = [RNG.choice(x, len(x)).mean() for _ in range(B)]
    return f"{x.mean():.2f} [{np.percentile(bs, 2.5):.2f}, {np.percentile(bs, 97.5):.2f}]"


def load(path):
    df = pd.read_csv(path)
    cols = {}
    for c in df.columns:
        m = re.match(r"\s*\[(W\d\d|P\d\d|C\d)\]", str(c))
        if m:
            cols[m.group(1)] = c
    missing = set(KEY) - set(cols)
    assert not missing, f"columns not found for {sorted(missing)}"
    long = []
    for i, row in df.iterrows():
        bg = {"lang": next((row[c] for c in df.columns if "language" in str(c).lower()), None),
              "games": next((row[c] for c in df.columns if "game" in str(c).lower()), None)}
        for qid, c in cols.items():
            a = str(row[c]).strip().upper()[:1] if pd.notna(row[c]) else None
            long.append({"pid": i, "id": qid, "answer": a, **bg})
    return pd.DataFrame(long)


def analyse(long, outdir):
    os.makedirs(outdir, exist_ok=True)
    out = ["# Human baseline: results", ""]
    # exclusion (preregistered): drop a participant who fails BOTH attention checks
    chk = long[long.id.isin(["C1", "C2"])].copy()
    chk["ok"] = chk.apply(lambda r: r.answer == KEY[r.id]["target"], axis=1)
    bad = chk.groupby("pid").ok.sum()
    excl = set(bad[bad == 0].index)
    d = long[~long.pid.isin(excl)]
    n = d.pid.nunique()
    out += [f"Participants: {long.pid.nunique()} responded, {len(excl)} excluded (failed both attention checks), "
            f"**{n} analysed**.", ""]

    # ---------- pointing ----------
    p = d[d.id.str.startswith("P")].copy()
    p["correct"] = p.apply(lambda r: r.answer == KEY[r.id]["target"], axis=1)
    p["nearest"] = p.id.map(lambda q: KEY[q]["target_is_nearest_hand"])
    per = p.groupby("pid").correct.mean()
    out += ["## H1. Pointing: people vs models on the same 12 images", "",
            f"- People (mean of participant accuracies, 95% bootstrap CI): **{boot(per)}**"]
    for split, lab in ((True, "target IS nearest the hand"), (False, "target NOT nearest the hand")):
        out.append(f"  - {lab}: {boot(p[p.nearest == split].groupby('pid').correct.mean())}")
    item_h = p.groupby("id").correct.mean()
    for m in ("gemma4:12b", "qwen3.5:9b", "gemma3:12b"):
        mc = pd.Series({q: MOD[q].get(m) == KEY[q]["target"] for q in item_h.index}).astype(float)
        diff = (item_h - mc).values
        obs = diff.mean()
        perm = np.mean([abs((diff * RNG.choice([-1, 1], len(diff))).mean()) >= abs(obs) for _ in range(B)])
        near = mc[[KEY[q]["target_is_nearest_hand"] for q in mc.index]].mean()
        far = mc[[not KEY[q]["target_is_nearest_hand"] for q in mc.index]].mean()
        out.append(f"- {m}: {mc.mean():.2f} (nearest-hand items {near:.2f}, others {far:.2f}); "
                   f"people minus model = {obs:+.2f}, item-level sign-flip p = {perm:.4f}")
    out += ["", "Per item (people's accuracy):", "", "```",
            pd.DataFrame({"people": item_h.round(2),
                          "sep": [KEY[q]["sep"] for q in item_h.index], "dist": [KEY[q]["dist"] for q in item_h.index],
                          "nearest_hand": [KEY[q]["target_is_nearest_hand"] for q in item_h.index],
                          **{m: [MOD[q].get(m) == KEY[q]["target"] for q in item_h.index]
                             for m in ("gemma4:12b", "qwen3.5:9b", "gemma3:12b")}}).to_string(), "```", ""]

    # ---------- whose left ----------
    w = d[d.id.str.startswith("W")].copy()
    w["label"] = w.apply(lambda r: KEY[r.id]["spot_labels"].get(r.answer, "invalid"), axis=1)
    w["facing"] = w.id.map(lambda q: KEY[q]["facing"]); w["rel"] = w.id.map(lambda q: KEY[q]["rel"])
    w["anchor"] = w.id.map(lambda q: KEY[q]["anchor"])
    hum = w.groupby(["rel", "facing"]).label.value_counts(normalize=True).unstack(fill_value=0).round(2)
    mrows = []
    for q, k in KEY.items():
        if k["study"] != "whose_left":
            continue
        for m, cnt in MOD[q].items():
            for lab, c in cnt.items():
                mrows += [{"rel": k["rel"], "facing": k["facing"], "model": m, "label": lab}] * c
    mod = pd.DataFrame(mrows).groupby(["rel", "facing"]).label.value_counts(normalize=True).unstack(fill_value=0).round(2)
    out += ["## H2/H3. Whose left? Which reading do people choose when the viewer's and the object's frames conflict?", "",
            "People:", "", "```", hum.to_string(), "```", "", "Models (all 5, all paraphrases, ambiguous wording):", "",
            "```", mod.to_string(), "```", ""]
    by_anchor = w[w.rel == "left"].groupby("anchor").label.value_counts(normalize=True).unstack(fill_value=0).round(2)
    out += ["'Left' items by anchor (people):", "", "```", by_anchor.to_string(), "```", ""]
    # H3: are individuals consistent? share of viewer-frame answers per participant on 'left' items
    lv = w[(w.rel == "left") & w.label.isin(["viewer", "object"])].groupby("pid").label.apply(lambda s: (s == "viewer").mean())
    cons = pd.cut(lv, [-0.01, 0.25, 0.75, 1.0], labels=["mostly object", "mixed", "mostly viewer"]).value_counts()
    out += ["Individual consistency on 'left' items (share of viewer-frame answers per person):", "",
            "```", cons.to_string(), "```", "",
            f"Agreement between people: for each 'left' item, the share choosing the majority reading: "
            f"{w[w.rel == 'left'].groupby('id').label.apply(lambda s: s.value_counts(normalize=True).iloc[0]).mean():.2f}", ""]
    # background splits (descriptive only)
    if p.lang.notna().any():
        out += ["## Background (descriptive only)", "", "```",
                p.groupby("lang").correct.mean().round(2).to_string(), "",
                w[w.rel == "left"].groupby("games").label.value_counts(normalize=True).unstack(fill_value=0).round(2).to_string(),
                "```"]
    open(os.path.join(outdir, "RESULTS_HUMAN.md"), "w", encoding="utf-8").write("\n".join(out) + "\n")
    print("\n".join(out))


def simulate(n):
    """Fake respondents, ONLY to test this script: 95% pointing accuracy, 60% viewer reading."""
    rows = []
    for i in range(n):
        r = {"Timestamp": i, "First language": RNG.choice(["Urdu", "Punjabi", "English"]),
             "Do you play 3D video games?": RNG.choice(["Often", "Sometimes", "Never"])}
        for q, k in KEY.items():
            col = f"[{q}] {k['question']}"
            if k["study"] in ("pointing", "check"):
                r[col] = k["target"] if RNG.random() < 0.95 else RNG.choice(list(k["letters"]))
            else:
                inv = {v: s for s, v in k["spot_labels"].items() if v != "neither"}
                r[col] = inv.get("viewer" if RNG.random() < 0.6 else "object", "A")
        rows.append(r)
    os.makedirs(os.path.join(HERE, "sim"), exist_ok=True)
    path = os.path.join(HERE, "sim", "fake_responses.csv")
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


if __name__ == "__main__":
    if sys.argv[1] == "--simulate":
        analyse(load(simulate(int(sys.argv[2]))), os.path.join(HERE, "sim"))
    else:
        analyse(load(sys.argv[1]), os.path.join(HERE, "results"))
