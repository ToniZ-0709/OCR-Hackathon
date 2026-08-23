# Review sheet

The canonical decisions are in `manual_labels.csv`. The no-human workflow is:

```powershell
python pipeline.py multi-agent-review
python pipeline.py manual-import
```

The command marks all 400 rows as `multi-agent-visual-audit`, applies the FMCG-only policy, and records the decision in `../reports/multi_agent_decision.json`.

The CSV remains editable for an optional later override:

- Use only `PRESENT` or `ABSENT` in `final_gate`.
- Leave `final_summary` empty for `ABSENT`.
- Keep `final_summary` grounded in visible evidence for `PRESENT`.

Run `python pipeline.py manual-status` to see progress.
