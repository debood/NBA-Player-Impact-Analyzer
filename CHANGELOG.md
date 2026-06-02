# Changelog

All notable project milestones are documented here.

## Version 3.1 — GitHub Cleanup and Portfolio Version

### Added

- Expanded README with project overview, methodology, version history, resume summary, skills demonstrated, and limitations.
- Cleaner GitHub project structure with app, data, notebooks, and docs separated.
- `requirements.txt` for easier local setup.
- `docs/resume_notes.md` with resume and LinkedIn wording options.

### Changed

- Reframed the project as a focused NBA Player Impact Analyzer instead of a general passion project repo.
- Moved Streamlit app code into `app/streamlit_app.py`.
- Moved cleaned player dataset into `data/processed/`.
- Moved raw data files into `data/raw/`.
- Moved notebook into `notebooks/` with a clearer filename.
- Reworked headshot logic to use `personId` from the final dataset instead of requiring `nba_api` in the app.

### Cleaned

- Removed unnecessary notebook/cache files from the final zip package.
- Added required-column checks to make debugging easier.
- Split repeated Streamlit logic into helper functions.

## Version 3.0 — Streamlit Matchup App

### Added

- Team A vs Team B lineup selection.
- Required starting five for each team.
- Optional bench players.
- Duplicate player warning.
- Projected final score and winner.
- Projected team box score.
- Projected player box score.
- Matchup breakdown chart for offense, defense, spacing, playmaking, and rebounding.
- Player cards with headshots, roles, and lineup scores.

## Version 2.0 — Player Profiles and Impact Score

### Added

- Player profile categories.
- Scoring value.
- Offensive creation value.
- Defensive impact value.
- Spacing value.
- Playmaking value.
- Rebounding value.
- Overall lineup score.
- Final cleaned player profile CSV.

## Version 1.0 — Notebook Exploration

### Added

- Initial NBA player analysis notebook.
- Early data cleaning and merging workflow.
- Exploratory player comparisons.
- First version of player impact logic.
