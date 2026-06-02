# Project Cleanup Notes

## What changed from the old repo

The original repo README described the project as a general passion project collection. The updated version is now focused on one clear product: an NBA Player Impact Analyzer built with Streamlit.

## Main cleanup decisions

- Renamed the project around the actual app purpose.
- Moved the Streamlit app into `app/streamlit_app.py`.
- Moved the final cleaned dataset into `data/processed/`.
- Moved raw supporting files into `data/raw/`.
- Moved the notebook into `notebooks/` with a clearer name.
- Added `requirements.txt` so the app can be installed and run more easily.
- Added `.gitignore` for Python, notebook, and environment files.
- Rewrote the README to explain what the app does, how the score works, and how to run it.

## Code cleanup decisions

- Removed the `nba_api` dependency from the app.
- Created headshot URLs directly from `personId`, which is already in the final dataset.
- Wrapped app logic in a `main()` function.
- Added a required-column check for easier debugging.
- Split repeated logic into helper functions.
- Kept optional bench players.
- Kept duplicate player warnings without blocking the app.
