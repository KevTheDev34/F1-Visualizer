# F1 Visualizer — Handoff

## Current goal
Build a Streamlit dashboard that clusters F1 drivers in a given race by behavioral signal (pace shape + per-compound tire degradation), names the clusters via Claude Haiku, and surfaces *standouts* — drivers whose behavior diverges from the field — rather than producing a clean field-wide taxonomy.

## Done
- Pipeline: FastF1 session load → clean-lap filter → per-driver feature matrix → StandardScaler → AgglomerativeClustering with dendrogram visualization.
- Per-compound tire degradation slopes folded into the feature matrix (field-median imputation for missing compounds; `min_stint_laps=10`).
- Cluster cards render three pace metrics (Median Pace, Pace Std Dev, Best Lap Delta) as `st.metric` with delta vs. field mean — Median Pace uses `delta_color="inverse"` so green = faster; the other two are neutral.
- Tire-degradation bars per cluster (seconds gained per lap of tire life).
- Caption above the cluster grid explains what the arrows/deltas mean and what positive/negative tire-deg bars represent.
- LLM cluster naming via Anthropic SDK (`claude-haiku-4-5-20251001`), cached with `@st.cache_data` keyed on the summary dict.
- Sidebar bug fix: Season selectbox lives outside the form so changing the year triggers an immediate rerun and refreshes the Race list (previously batched inside the form, which suppressed reruns until submit).
- End-to-end runs verified across multiple race weekends through the LLM step.

## In progress
Nothing actively in flight. Last shipped commit on `origin/main` is `8f3bc91 Fix stale race dropdown and document cluster-card metrics`.

## Next (backlog, roughly priority-ordered)
- **Behavior-over-pace features**
  - DRS telemetry features (usage patterns, overtaking signal)
  - Race-craft features derived from Position deltas (places gained/lost, recovery laps)
- **Visualization**
  - Cluster-discriminating track-segment view (which corners/sectors separate the clusters)
- **Modeling refinements**
  - Fuel-correction for tire degradation slopes so the regression isn't conflating fuel burn-off with tire wear
  - Cross-season same-circuit comparison (e.g., 2023 vs. 2024 vs. 2025 Monza)
- **Robustness**
  - `qcut` guard for races with <3 viable clusters
  - Empty `tire_profile` handling (e.g., fully wet races where a compound never ran)
  - Singleton-cluster unification (decide: merge to nearest, or render as "outlier")
  - Audit `describe_cluster` docstring against current feature set
  - `@st.cache_data` on `get_session_data` to skip re-loading the same session within a Streamlit run

## Open questions
- Singleton clusters: are they signal (a true standout, which is the point of the project) or noise (merge into nearest)? Current behavior is to leave them as-is; needs a deliberate decision once DRS/race-craft features come in and change cluster shapes.
- Fuel correction: assume a linear mass-vs-laptime coefficient from public estimates, or skip and document the bias? Affects how the tire-deg bars should be interpreted.
- Cross-season comparison: same dashboard with a multi-select, or a separate page? Affects how `get_session_data` is structured.
- LLM cost: cache is keyed on the summary dict, but the dict changes any time features change. Worth persisting cluster names to disk by `(year, race, feature_hash)`?

## Key files
- `main.py` — the entire app: session loading, lap filtering, feature engineering, scaling, clustering, dendrogram, cluster cards, LLM naming, sidebar.
- `requirements.txt` — `fastf1==3.8.3`, `streamlit==1.57.0`, `scikit-learn==1.8.0`, `pandas==2.3.3`, plus unpinned `anthropic` and `python-dotenv`.
- `.env` — holds `ANTHROPIC_API_KEY` (gitignored; never read or echo).
- `cache/` — FastF1 disk cache (gitignored).
- `~/.claude/projects/-Users-kevynramirez-Documents-Projects-F1-Visualizer/memory/`
  - `clustering_targets_standouts.md` — project intent: surface divergent behavior, use field-median imputation, not driver-dropping.
  - `fastf1_vpn_gotcha.md` — partial-load failures (`DataNotLoadedError` after a "successful" `session.load()`) are often a VPN issue, not cache corruption. Check VPN state before suggesting cache deletion.
