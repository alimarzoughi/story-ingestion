# Seed evaluation set

`seed.csv` is hand-labelled. Target ~120 rows before the first live run:

| kind | rows | expected_same_story | what it tests |
|---|---|---|---|
| same_story | ~40 | true | two outlets, same development → must attach to one story |
| hard_negative | ~40 | false | same topic / same actors, different development → must NOT merge |
| opinion_attach | ~20 | true | opinion/analysis attaches but never creates a story |
| syndication | ~20 | true | wire copies attach but are excluded from pairs |

Source for rows: the legacy `articles` table (months of Fox/HuffPost/NYP/Vox rows). Pull ~200 recent titles with
`select url,title,source_name,published_at from articles order by published_at desc limit 200`, pair them by hand.

`python -m pipeline.cli eval` (to be added in Phase 2) ingests every URL in this file, enriches, assigns, and reports
precision/recall of the attach decision at the configured thresholds.
