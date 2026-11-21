"""Score a `recheck-stories --report` TSV against hand labels.

    python eval/score_gold.py local/recheck-dry.tsv eval/gold_nvidia_2026-11-20.tsv

Gold labels (gold_nvidia_2026-11-20.tsv, from the GPT 6.2 Astra audit of 2026-11-20, one editor's judgment):
  On event   the headline is about the story's own development (report or focused reaction)
  Related    plausible background link, weak as a direct comparison
  Separate   an independently identifiable development (belongs in another story)
  Mixed      several topics, or ambiguous
The pipeline's answer for each rechecked member: detached -> Separate; kept as direct_report / direct_reaction ->
On event (shown); kept as related -> Related; kept as mixed -> Mixed (both hidden). Story founders are never
rechecked, so they are not in the report and are left out of the score.

Known disagreement: Astra counts "board weighs ousting Huang" as Separate from "Huang resigns"; Ali decided
(2026-11-20) that they are one story, so those rows are expected to come out On event.
"""
from __future__ import annotations

import csv
import sys
from collections import Counter

LABELS = ("On event", "Related", "Mixed", "Separate")


def pipeline_label(row: dict) -> str:
    if row["decision"] == "detached":
        return "Separate"
    return {"related": "Related", "mixed": "Mixed"}.get(row.get("role") or "", "On event")


def main(report_path: str, gold_path: str) -> int:
    with open(gold_path, encoding="utf-8", newline="") as fh:
        gold = {r["article_id"]: r for r in csv.DictReader(fh, delimiter="\t") if r["label"] in LABELS}
    with open(report_path, encoding="utf-8", newline="") as fh:
        report = {r["article_id"]: r for r in csv.DictReader(fh, delimiter="\t")}
    both = [a for a in report if a in gold]
    if not both:
        print("no labelled article appears in the report")
        return 1
    pairs = Counter((gold[a]["label"], pipeline_label(report[a])) for a in both)
    agree = sum(n for (g, p), n in pairs.items() if g == p)
    print(f"{len(both)} labelled articles in the report; exact agreement {agree}/{len(both)} ({agree / len(both):.0%})\n")
    print("gold \\ pipeline".ljust(16) + "".join(p.rjust(11) for p in LABELS))
    for g in LABELS:
        print(g.ljust(16) + "".join(str(pairs[(g, p)]).rjust(11) for p in LABELS))

    shown = lambda a: pipeline_label(report[a]) == "On event"
    wrong_shown = [a for a in both if gold[a]["label"] in ("Separate", "Related", "Mixed") and shown(a)]
    lost = [a for a in both if gold[a]["label"] == "On event" and not shown(a)]
    print(f"\nshown but not on the event (what readers would still see wrongly): {len(wrong_shown)}")
    for a in wrong_shown[:25]:
        print(f"  [{gold[a]['label']}] {gold[a]['outlet']}: {gold[a]['headline'][:90]}")
    print(f"\non the event but hidden or detached (coverage lost from the story): {len(lost)}")
    for a in lost[:25]:
        print(f"  [{pipeline_label(report[a])}] {gold[a]['outlet']}: {gold[a]['headline'][:90]}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1], sys.argv[2]))
