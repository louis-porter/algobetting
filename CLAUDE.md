# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A sports betting prediction system for football/soccer, focused on developing statistical models to find profitable edges in sportsbook markets. This is an active work-in-progress.

## Environment Setup

Python 3.13.7 virtual environment at `.venv/`:

```bash
source .venv/bin/activate
```

No package manager config (no pyproject.toml/setup.py). Dependencies are managed directly in the venv.

## Running Scripts

Scripts are run directly — there is no build/test/lint system:

```bash
# Run the main data collection orchestrator
python infra/data/collectors/_collect_all_strength.py

# Run individual scrapers
python infra/data/collectors/fotmob_season_downloader.py
python infra/data/collectors/whoscored/main.py
```

Models live in Jupyter notebooks under `algo/models/` — open with `jupyter notebook`.

## Architecture

```
Data Scraping (infra/data/collectors/)
    ↓
Raw Storage (infra/data/json/, infra/data/db/fotmob.db ~748MB)
    ↓
Feature Engineering (infra/data/feature_engineering/)
    ↓
Model Notebooks (algo/models/)
    ↓
Outputs (output/*.csv)
```

### Data Collectors

Five independent scrapers feed the pipeline:
- **WhoScored** (`whoscored/main.py`) — Selenium-based; collects match events, shots, passes with EPV (Expected Possession Value)
- **FotMob** (`fotmob_season_downloader.py`) — Threaded JSON API fetcher using `curl_cffi` for browser impersonation; stores into SQLite (`fotmob.db`)
- **FBRef** (`fbref_*.py`) — Selenium-based; match summaries and advanced stats
- **Understat** (`understat_shot_scraper.py`) — API-based shot data
- **ClubElo** (`clubelo_*.py`) — Team Elo strength ratings

`_collect_all_strength.py` orchestrates the FotMob + WhoScored pipeline together.

### Feature Engineering

`infra/data/feature_engineering/param_optimisation.py` — key concepts:
- Exponential decay weighting over time (default decay_rate=0.0077, 180-day window)
- Recent game weighting
- Red card impact modeling (30% weight reduction)
- Parameter optimization via SciPy with XGBoost as the objective

### Models (`algo/models/`)

- **Team Strength**: Bayesian multilevel models (PyMC), Dixon-Coles parametric, non-penalty variants
- **XGBoost regressors/classifiers**: team goals, team shots, team corners (over/under 11.5), player shot outcomes
- **Linear baselines** for each target

Model traces are saved locally in `model_traces/` subdirectories within each model folder.

### Key Dependencies

| Purpose | Libraries |
|---|---|
| Scraping | selenium, beautifulsoup4, curl_cffi, requests |
| Data | pandas, numpy, scipy |
| ML | xgboost, scikit-learn |
| Bayesian | pymc, arviz, blackjax |
| Performance | numba (JIT for weighted calculations) |

### Data Sources → Feature Mapping

WhoScored and FotMob use different team naming conventions. `infra/data/collectors/whoscored/team_mapping.py` bridges the two systems.

Shot sequences and dependent shot logic (recent commits) are used to model within-match shot correlations.
