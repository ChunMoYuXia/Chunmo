# 实施决策记录

日期：2026-10-15。以下保留全部 54 项裁决、原因与误判代价，按当时上下文记录。原生工具链的历史阻塞已在用户授权后解除，当前状态见文末和验收记录。技术记录沿用开发时原文；功能范围以验收记录及 FR 表为准。

1. Ruling: Keep the user-requested dedicated new repository at D:/workplace/stock-portfolio-client and use a feature branch instead of moving it to another managed worktree — this repository contains no existing product checkout and the approved design fixes its location — cost if wrong: worktree placement only, no existing product changes.

2. Ruling: Use bundled Python 3.12 in the project .venv rather than global Python 3.14 — stable packaging and numerical dependency compatibility — cost if wrong: recreate project environment; Python floor documented as 3.12.

3. Ruling: Native Bash sdd-workspace failed on protected hidden-directory writes and entered a retry loop; use the same plan-scoped ledger/brief structure with PowerShell/Python bookkeeping — directory initialized through authorized escalation — cost if wrong: bookkeeping scripts only, Git commits remain authoritative.

4. | 5→6 | FeeSchedule and rebalance cost integration | Ruling: finish optimizer first, complete rebalance accounting after Task6 to avoid circular dependency; cost if wrong: implementation order only |

5. Task 1: Ruling: Browser reload uses a session-only HttpOnly SameSite=Strict cookie initialized from a one-time fragment launch token; Tauri keeps header authentication — no token in web storage and cookie writes require a trusted Origin — cost if wrong: browser CSRF/session boundary, covered by explicit tests.

6. Task 1: Ruling: pytest uses unique project-local .cache/pytest directories because the shared Windows TEMP tree is inaccessible; add httpx2 as test dependency for current Starlette compatibility — cost if wrong: test environment only, production HTTPX provider client unchanged.

7. Task 2: Ruling: Real CSV imports require a supplied calendar; daily metadata changes are stored at their observation availability time instead of collapsing to a future master row — prevents missing days/metadata from silently creating historical assumptions — cost if wrong: user must supply calendar and historical metadata gaps remain explicitly declared.

8. Task 3: Ruling: Separate read-only market_queries.py/quotes.py from dataset writes and add optional start/lookback to the market port — bounded per-security history avoids reloading full datasets for interactive features — cost if wrong: future provider adapters must implement the documented optional query arguments.

9. Task 3: Ruling: FR-213 complete acceptance belongs to M1 because aliases/concepts are beyond the approved M0 code/name/initials/industry search — retain earliest M0 and honest partial status — cost if wrong: the additional search dimensions remain explicitly scheduled for M1.

10. Task 4: RED: missing DSL/API behaviors, then exact 20% boundary excluded by float subtraction. GREEN: 46 backend tests, Ruff and strict mypy. Ruling: compute endpoint momentum ratio with Decimal before conversion to float and share API market DTOs in market_schemas.py — threshold precision and route independence — cost if wrong: one metric/DTO refactor; golden boundary test covers it.

11. Task 5: RED: 11 missing optimizer/API failures, then short requested window silently accepted. GREEN: 58 tests, Ruff and strict mypy passed. Targets and imported holdings persist across restart. Rebalance integration remains assigned to Task6 by pre-flight ruling. Ruling: scale covariance objective by largest variance and floor API weights to 12 decimals — SLSQP tolerance must not accept initial weights merely due to small daily variances, and serialization must not create leverage — cost if wrong: sub-picoweight cash residual, explicitly retained.

12. Task 6: Ruling: predefined statutory schedules start 2022-04-29, broker commission is illustrative and includes regulatory fees; gross dividends exclude holding-period tax adjustment, bonus fractions floor with a ledger note and pay_date defines delivery — these require broker/registrar-specific data to be exact — cost if wrong: historical or tax-adjusted results need supplied fee/action rules and are explicitly qualified.

13. Task 6: Ruling: automatic execution currently accepts Shanghai/Shenzhen 100-share lot rules; STAR and Beijing order-unit rules are rejected explicitly instead of applying an incorrect generic lot — the M0 lot_sizes contract lacks separate minimum/increment semantics — cost if wrong: those markets require a later dedicated rule adapter; research optimization remains available.

14. Task 7: Ruling: use conservative opening-only day orders; opening at upper/lower limit blocks the corresponding side even if the later daily range unlocks it — daily bars cannot establish queue position or intraday chronology without hindsight — cost if wrong: execution may understate fills; assumptions and unfilled reasons expose it.

15. Task 7: Ruling: extend rebalance with optional receivables and pending_holdings; include their value in target NAV but never spend receivables or sell undelivered shares — Task6 introduces assets absent from the original cash/holdings signature — cost if wrong: an additional keyword contract for future adapters, defaults preserve existing callers.

16. Task 8: Ruling: retain the Task7 NavPoint.nav field rather than brief shorthand value; the input-only metrics DTO translates its value field explicitly — avoid breaking the established producer — cost if wrong: one documented DTO mapping.

17. Task 8: Ruling: full-cash exposures show cash_weight=1 while historical VaR/CVaR remain null(all_cash_no_market_returns); distinguish actual holdings from hypothetical fixed target weights and require every benchmark session — avoids manufacturing risk history or changing comparison horizon silently — cost if wrong: users must choose target mode or supply complete history to obtain those estimates.

18. Task 9: Ruling: require Idempotency-Key for every financial command including create/advance/halt/resume/cancel, with global create scope and per-account operation scopes — the canonical spec is broader than the order-only brief example — cost if wrong: clients must retain a key across retries; generated UI handles it.

19. Task 9: Ruling: persist drained AccountState events as paginated ledger rows and retain ledger_offset; store compact command responses and verify materialized account/order/trade/ledger rows against serialized audit transitions in one read snapshot — avoids quadratic historical response storage and hash-only false reassurance — cost if wrong: ledger projection and audit schema require coordinated future migration.

20. Task 9: Ruling: account defaults to the 21st available session, accepts explicit calendar start, and market-order reservation uses a disclosed 20% buffer with fill-time affordability recheck — ensures initial research history is usable without looking at the future execution open — cost if wrong: market orders can require more reserved cash or fill fewer shares; limits are available.

21. Task 10: Ruling: eagerly call the public app.openapi() at composition completion and construct test apps sequentially while keeping SQL requests concurrent — installed FastAPI lazily rebuilds dependency fields on first routing, and strict concurrent requests reproduced UnsupportedFieldAttributeWarning even after sequential app construction — cost if wrong: startup schema compilation time; no HTTP documentation or secret exposure.

22. Task 10: Ruling: use one worker with at most16 active durable jobs; retain immutable result artifacts and compact list summaries, and add selected raw-price benchmark plus explicit annualization/risk-free parameters — local research stays bounded and benchmark comparison needs real aligned data — cost if wrong: long jobs queue serially; total-return benchmark needs richer data in a later stage.

23. Task 11: Ruling: staged navigation exposes only implemented routes; screening-to-portfolio transfer is connected in Task 12 with the real destination — avoids shipping links to nonexistent pages — no final scope loss.

24. Task 12: Ruling: browser session storage persists pending financial command path/body/key (no credentials), never automatically replays — retries after reload can reuse the original attempt — storage-disabled browsers retain attempts only in memory and must inspect durable orders after reload.

25. Task 12: Ruling: benchmark remains raw price on a separately labelled chart axis; performance comparisons use backend metrics — avoids confusing price with portfolio equity — chart is not a normalized relative-performance comparison.

26. Task 13: Ruling: provider port yields vendor-neutral text/usage chunks; application publishes versioned AIEvent only after complete paragraph validation — domain orchestration owns run/sequence/cancellation, vendor adapter owns wire protocol — raw token typing is deliberately not exposed.

27. Task 13: Ruling: rules report is deterministically reconstructed from durable research artifact with rule-report-v1, no separate mutable report table — avoids duplicating research truth — later report edition retention needs versioned persistence.

28. Task 13: Ruling: numeric guard is conservative, rejects Unicode/Chinese numbers and common implicit multiples outside trusted metric tokens — prevents unsupported quantitative text — ordinary phrases containing numeral characters can be rejected, documented in docs/ai.md.

29. Task 13: Ruling: optional AI uses backend environment configuration and explicit max_completion_tokens/include_usage compatible protocol; no silent provider/model/protocol retry — secrets stay outside UI and repeated charges stay explicit — providers with older incompatible APIs require their own adapter. Real provider remains unconfigured/unverified.

30. Task 14: Ruling: the Python distribution uses PyInstaller onedir plus fixed Tauri resources, rather than onefile subprocess extraction — guarantees one directly owned server process and bounded shutdown — distribution is ~151 MB before ZIP compression.

31. Task 14: Ruling: every CLI service acquires a data-directory OS lock before schema/job recovery; native parent pipe EOF requests graceful shutdown — prevents competing instances from recovering another live worker — second clients for the same data directory are rejected.

32. Task 14: Ruling: Rust 1.99.0 components were downloaded from the official manifest, SHA256 verified and extracted only to project .cache; Cargo.lock resolved 414 dependencies and rustfmt ran. cargo check --locked actually failed because MSVC link.exe is absent; current user is not an administrator. No global compiler / Windows SDK installation was attempted — native type compilation/window/NSIS acceptance remains blocked, portable browser client is a separately verified output.

33. Task 14: Ruling: reject future/foreign database versions before WAL or DDL; offline backup takes the service lock and copies a SQLite snapshot plus raw sources — preserves future data and provenance — future released migrations need an explicit new version implementation.

34. Task 14: Ruling: catch unexpected handler errors inside the HTTP boundary before Starlette re-raises to Uvicorn — a real-process SQL fault reproduced private parameter leakage and the regression now checks sanitized trace-only diagnostics — streaming errors after headers remain constrained by their own typed/event boundaries.

35. Task 14: Ruling: create a verified local candidate commit before the one whole-branch review, so review-package has a stable commit range; review fixes remain local before handoff — no merge or publication — cost if wrong: additional local commit only. Native compilation is explicitly unverified and stays a separate acceptance gate.

36. Final: Ruling: Tauri compilation/window/NSIS set aside — preserve explicit failed-MSVC gate and require actual system-toolchain approval before native acceptance — cost if wrong: native output stays unavailable; portable output remains independently verified.

37. Final: Ruling: real AI compatibility/billing/prose set aside — no provider credentials or real inference, only controlled adapter/guard evidence — cost if wrong: user must test chosen provider before relying on its output.

38. Final: Ruling: live data/broker/profitability/full-market performance set aside — M0 delivers local research with explicit synthetic provenance and one bounded benchmark — cost if wrong: these remain unsupported uses, not promised outcomes.

39. Final: Ruling: M1–M3 features set aside — approved phased scope and124-FR partial/unimplemented tracking stand — cost if wrong: future phases require their own implementation and acceptance.

40. Final: Ruling: opening-only execution/fixed weights/gross tax/STAR-BSE exclusions set aside — retain disclosed conservative model rather than imply market fidelity — cost if wrong: estimates differ from actual execution/tax; unsupported markets stay rejected.

41. Final: Ruling: changed-payload retry policy set aside — explicit warning and durable-order inspection stand; exact-payload retry reuses its key while modified intent is a new command — cost if wrong: a user can manually create duplicate intent after an uncertain response.

42. Final: Ruling: extreme finite magnitudes/prolonged SQLite contention set aside — no demonstrated ordinary-input failure in this review; retain finite validation and bounded local workload claims — cost if wrong: adversarial extremes may need further limits and lock diagnostics.

43. Final: Ruling: remote CI/publication set aside — user deferred repository selection; workflow configuration is unexecuted evidence — cost if wrong: first remote run may expose environment differences.

44. Final: Ruling: do not silently rewrite old misidentified920 datasets — preserve immutable provenance and require reimport; defense-in-depth execution guard rejects920 independent of old exchange metadata — cost if wrong: old research IDs remain mislabeled until reimported, but cannot auto-execute.

45. Final: Ruling: replace only two fixed build-owned runtime directories after resolved-path containment/link checks; no merge-copy — actual repeat-build verification exposed stale files — cost if wrong: failed copy requires rebuilding generated output; user data and unrelated artifacts are outside allowed targets.

46. Final: Ruling: initialize backtest dates only when portfolio/dataset/calendar identity or bounds change — record refresh must preserve in-progress parameters — cost if wrong: same-ID mutable records would need explicit revision, whereas current saved portfolios/datasets are immutable.

47. Final: Ruling: bump CSV parser tocsv-v2 and include parser version in content identity — otherwise dedup could reuse legacy wrong920 metadata on identical reimport — cost if wrong: identical file parsed under a new version creates a new immutable dataset rather than deduplicating across parser semantics; regression RED→GREEN.

48. Native: Ruling: install the independently signed Microsoft C++ Build Tools/SDK only after explicit user authorization; VSCode remains a valid editor — native linker prerequisites are independent of the editor — cost if wrong: system disk/tool maintenance; no automatic reboot performed.

49. Native: Ruling: seed project-local NSIS3.13/plugins after official SHA256 + Tauri SHA1 verification when built-in network fetch timed out — same official toolchain, no alternate mirror — cost if wrong: Tauri hash checks reject mismatched tools; cache can be rebuilt.

50. Native: Ruling: rename internal library target toportfolio_research_lib — Cargo reported binary/library PDB filename collision — cost if wrong: internal crate import requires corresponding rename; main.rs and real builds verify it.

51. Native: Ruling: validate actual current-user installation with process-local WebView debugging and a project-local WebView profile, then close its real window — enables observable IPC/origin/cleanup tests without persistent debugging settings — cost if wrong: this verifies this Windows/WebView2 environment only, not every OS/runtime version.

52. Native final: Ruling: compare installed and bare Tauri executables after normalizing only the unique __TAURI_BUNDLE_TYPE_VAR_NSS marker to __TAURI_BUNDLE_TYPE_VAR_UNK; require identical length and every other byte — NSIS deliberately records package type, so whole-file hash equality is the wrong acceptance check — cost if wrong: an overly broad normalization could mask modified code; this check permits exactly the observed unique three-byte marker difference.

53. Native final: Ruling: remove only 18 identified obsolete generated JavaScript files from the resolved installation resources/backend/_internal/web/_app/immutable subtree, then require the exact current 434-file runtime set and hashes — repeated acceptance installs retained prior hashed assets — cost if wrong: deleting unknown files could destroy user content; each removed file was a build-owned JavaScript artifact, and user workspace/data were outside the cleanup target.

54. Native final: Ruling: document Codex MSIX LocalAppData redirection and launch the resolved physical installed executable for native acceptance — the logical default install path resolves into this app package's LocalCache/Local — cost if wrong: native installation evidence is specific to this Windows/MSIX/WebView2 environment; a normal shell installation path can differ.

## 最终验证

一次独立全分支评审，无 Critical、两项 Important 已先失败后通过修复；交付核对追加修复也已验证。后端164、前端45、源码E2E3、冻结E2E3全部通过；Rust格式 / 类型 / 发布构建、NSIS安装与真实WebView2窗口验收通过，关闭后自有后端和监听端口退出。安装程序包类型标记以外的代码字节和全部434个运行时文件已核对。没有延后 Minor。详见 [评审记录](code-review.md) 和 [验收记录](acceptance.md)。

保留本地功能分支，尚未上传 GitHub。真实AI未配置，M1–M3按批准阶段保留。
