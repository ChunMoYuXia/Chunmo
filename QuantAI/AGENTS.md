# Project development guide

- Read the approved design and current M0 plan in `docs/superpowers/` before changes.
- Keep domain rules independent of FastAPI, SQLAlchemy, UI and model vendors.
- API handlers validate DTOs and call application services; they do not implement trading rules.
- Share accounting and execution rules between backtests and paper trading.
- Use Decimal for money, raw prices for trades, explicit calendars and time-aware data versions.
- Never replace unknown values with zero or present synthetic data as live prices.
- Keep components and modules focused; prefer explicit typed interfaces over generic dictionaries.
- Write tests for domain invariants and failure boundaries. Use independently derived expected values.
- Run backend tests/Ruff/mypy and client check/lint/tests/build for relevant changes.
- Run backend with `.venv/Scripts/python`; test temporary files stay in `.cache/pytest`.
- Client commands run in `apps/client`; generated API types come from `contracts/openapi.json`.
- Keep secrets, real holdings, licensed data, caches and environment directories outside Git.
- Record partial requirements honestly in `docs/fr-traceability.md` and final evidence in `docs/acceptance.md`.
- This directory is the initial M0 source archive. The user authorized publishing this archive to ChunMoYuXia/Chunmo on 2026-10-15. Subsequent merged development remains local pending user acceptance; do not publish or package the merged version without new authorization.
