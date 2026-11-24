"""Build one short human survey that covers two pilot studies, plus the models' answers on exactly the same items.

  W01-W16  "Whose left?": the 16 frame-conflict items already prepared in frame-of-reference-3d/survey/
  P01-P12  "Bring me that one": 12 pointing scenes, one per cell of
           angle to closest rival (8/15/30 deg) x distance (near/far) x target-is-nearest-the-hand (yes/no)
  C1, C2   two easy pointing scenes used only as attention checks (preregistered exclusion rule)

Writes items/ (images), answer_key.json, model_answers.json and FORM_TEXT.md. CPU only.
"""
import json, math, os, shutil, sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FOR, PNT = os.path.join(ROOT, "frame-of-reference-3d"), os.path.join(ROOT, "that-one-pointing")
ITEMS = os.path.join(HERE, "items")
MODELS3 = ["gemma4:12b", "qwen3.5:9b", "gemma3:12b"]


def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def near_hand(s):
    return min(s["objects"], key=lambda o: math.dist(o["pos"], s["hand"]))["label"]


def main():
    os.makedirs(ITEMS, exist_ok=True)
    key, models = [], {}
    # --- Whose left? ---
    wkey = json.load(open(os.path.join(FOR, "survey", "answer_key.json")))
    place = [r for r in jl(os.path.join(FOR, "data", "placements.jsonl")) if r["frame"] == "ambiguous"]
    for k in wkey:
        qid = f"W{k['q']:02d}"
        shutil.copy(os.path.join(FOR, "survey", f"q{k['q']:02d}.png"), os.path.join(ITEMS, f"{qid}.png"))
        key.append({"id": qid, "study": "whose_left", "question": k["question"], "anchor": k["anchor"],
                    "facing": k["facing"], "rel": k["rel"], "spot_labels": k["spot_labels"]})
        rows = [r for r in place if (r["anchor"], r["facing"], r["rel"]) == (k["anchor"], k["facing"], k["rel"])]
        models[qid] = {m: dict(Counter(r["label"] for r in rows if r["model"] == m)) for m in sorted({r["model"] for r in rows})}
    # --- pointing ---
    scenes = jl(os.path.join(PNT, "data", "scenes.jsonl"))
    ans = [r for r in jl(os.path.join(PNT, "data", "answers.jsonl")) if r["cond"] == "image"]
    chosen, used = [], set()
    for sep in (8, 15, 30):
        for dist in ("near", "far"):
            for nh in (True, False):
                s = next(s for s in scenes if s["sep"] == sep and s["dist"] == dist and (near_hand(s) == s["target"]) == nh)
                chosen.append(s); used.add(s["id"])
    checks = [s for s in scenes if s["sep"] == 30 and s["dist"] == "near" and s["n"] == 3
              and near_hand(s) == s["target"] and s["id"] not in used][:2]
    for i, s in enumerate(chosen + checks):
        qid = f"P{i + 1:02d}" if i < len(chosen) else f"C{i - len(chosen) + 1}"
        shutil.copy(os.path.join(PNT, "data", "img", f"{s['id']}.png"), os.path.join(ITEMS, f"{qid}.png"))
        key.append({"id": qid, "study": "pointing" if qid.startswith("P") else "check", "scene": s["id"],
                    "question": "Which object is the person pointing at?", "letters": "ABCDEF"[:s["n"]],
                    "target": s["target"], "rival": s["rival"], "near_hand": near_hand(s), "sep": s["sep"],
                    "dist": s["dist"], "n": s["n"], "target_is_nearest_hand": near_hand(s) == s["target"]})
        models[qid] = {r["model"]: r["answer"] for r in ans if r["id"] == s["id"]}
    json.dump(key, open(os.path.join(HERE, "answer_key.json"), "w"), indent=1)
    json.dump(models, open(os.path.join(HERE, "model_answers.json"), "w"), indent=1)
    # --- form text (order: checks are mixed in; Google Forms can also shuffle question order) ---
    order = [k["id"] for k in key if k["study"] == "pointing"][:6] + ["C1"] + \
            [k["id"] for k in key if k["study"] == "whose_left"] + ["C2"] + \
            [k["id"] for k in key if k["study"] == "pointing"][6:]
    byid = {k["id"]: k for k in key}
    lines = ["# Google Form text (one image + multiple-choice question per item; keep the [ID] at the start of each title, the analysis uses it; turn on 'Shuffle question order')", "",
             "**Title:** Point and place: a 4-minute survey on everyday spatial language", "",
             "**Intro:** This is part of a small research project comparing how people and AI systems understand "
             "pointing and words like 'left' and 'in front of'. Go with your first instinct; there are no trick "
             "questions. The survey is anonymous: no names, emails or other identifying details are collected. "
             "You can stop at any time. By continuing you agree that your anonymous answers may be used in research.",
             "", "**Optional background:** First language (Urdu / Punjabi / English / Pashto / Sindhi / Other); "
             "Do you play 3D video games? (Often / Sometimes / Never)", ""]
    for n, qid in enumerate(order, 1):
        k = byid[qid]
        opts = "A, B, C or D" if k["study"] == "whose_left" else ", ".join(k["letters"])
        q = k["question"].split(" Which spot")[0] if k["study"] == "whose_left" else k["question"]
        lines.append(f"{n}. Image `items/{qid}.png`. Question title: \"[{qid}] {q}\" Options: {opts}")
    open(os.path.join(HERE, "FORM_TEXT.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print(f"{len(key)} items ({sum(k['study'] == 'whose_left' for k in key)} whose-left, "
          f"{sum(k['study'] == 'pointing' for k in key)} pointing, {sum(k['study'] == 'check' for k in key)} checks)")


if __name__ == "__main__":
    main()
