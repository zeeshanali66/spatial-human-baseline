# Preregistration: people vs vision/language models on pointing and "whose left?"

*Written 2026-11-25, before any human data was collected. Model answers were collected earlier (pilot studies
[that-one-pointing-vlm](https://github.com/zeeshanali66/that-one-pointing-vlm) and
[whose-left-llm-frame-of-reference](https://github.com/zeeshanali66/whose-left-llm-frame-of-reference)).*

## Background
In two pilots, open models (1) mostly picked the object **nearest the hand** instead of the one a person points at,
and (2) when "left of the sofa" was ambiguous, chose the viewer's or the object's left in a way that depended on the
wording. A model's behaviour is only "wrong" relative to what people do, so this survey measures people on **exactly
the same images and sentences**.

## Materials
30 items in one anonymous online form (about 4 minutes), question order shuffled by the form:
- **P01-P12, pointing:** one top-down scene per cell of angle to the closest rival object (8 / 15 / 30 deg) x target
  distance (near / far) x whether the target is also the object nearest the hand (yes / no). "Which object is the
  person pointing at?"
- **W01-W16, whose left:** "Put the lamp to the left of / in front of the sofa" with four marked spots, for 4 anchors
  (sofa, car, armchair, person) facing toward the viewer or sideways. Items where the two frames point to different
  spots are the test items; "in front of" + facing toward is a control where both frames agree.
- **C1, C2, attention checks:** two easy pointing scenes (30 deg, near, 3 objects).
- Optional: first language; plays 3D video games (often / sometimes / never). No names, emails or IDs are collected.

## Hypotheses
- **H1 (pointing).** People are accurate (mean >= 0.90) and their accuracy does **not** depend on whether the target is
  nearest the hand (difference < 0.10). People are more accurate than each model on the same 12 items
  (item-level sign-flip permutation test, Holm-corrected over 3 models).
- **H2 (whose left).** On conflicting "left" items, people do **not** agree on one reading: the majority reading is
  chosen by less than 80% of people per item on average. (Models chose the viewer's left almost every time with the
  ambiguous wording, so a split among people means the models are more confident than people are.)
- **H3 (exploratory).** People, like the models, use different frames for "in front of" and "left of"; and some
  individuals switch frames across items ("mixed" readers).

## Sample and stopping rule
Target 40 analysed participants. Collection stops at 60 responses or after three weeks, whichever comes first. All
questions are required, so there is no missing data.

## Exclusions
A participant who fails **both** attention checks is excluded. No other exclusions.

## Analysis
`analyze.py` (written and tested on simulated data before collection; the simulated output is not a result):
bootstrap 95% CIs over participants (5000 resamples); item-level sign-flip permutation tests; Holm correction.
Background variables are reported descriptively only.

## Ethics
Anonymous, minimal-risk survey of everyday spatial judgments with adult volunteers; no identifying data. Participants
are told the purpose and can stop at any time. If an institutional ethics review is required for publication, it will
be obtained before collection.
