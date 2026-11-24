# Human baseline for two spatial-language pilots (ready to run)

One short anonymous survey (30 items, ~4 minutes) that asks people exactly what the models were asked in
[that-one-pointing-vlm](https://github.com/zeeshanali66/that-one-pointing-vlm) (which object is the person pointing
at?) and [whose-left-llm-frame-of-reference](https://github.com/zeeshanali66/whose-left-llm-frame-of-reference)
(where is "left of the sofa"?), so model behaviour can be judged against human behaviour on the same items.

- `PREREGISTRATION.md`: hypotheses, sample, exclusions and analysis, fixed before data collection.
- `make_items.py`: builds `items/` (30 images), `answer_key.json`, `model_answers.json` (the models' answers on the
  same items, from the pilots) and `FORM_TEXT.md` (text to paste into a Google Form).
- `analyze.py`: reads the form's CSV export and writes `results/RESULTS_HUMAN.md`. Tested on simulated answers with
  `python analyze.py --simulate 40` (writes to `sim/`, not a result).

Status: materials ready; data collection not started.

## Disclosure
Code and documentation were written with the help of an AI assistant; the research design, checking and decisions
are the author's.
