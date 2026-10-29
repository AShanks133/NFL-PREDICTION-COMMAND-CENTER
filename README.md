# NFL Prediction Command Center (2026)

An interactive Streamlit dashboard with weekly NFL predictions, a configurable 1,000–100,000-run Monte Carlo matchup simulator, CSV downloads, and a walk-forward historical backtest.

## Install and run

1. Install Python 3.10+.
2. Unzip this project and open a terminal in the `nfl_sim` folder.
3. Run `pip install -r requirements.txt`.
4. Run `streamlit run app.py`.
5. Open the local address Streamlit prints (typically http://localhost:8501).

The app fetches 2026 regular-season scores and upcoming schedules from ESPN's public scoreboard endpoint when opened and every 30 minutes. Internet access is required. Use **Refresh feed** to reload.

## Tabs
- **Weekly predictions:** Choose a week; simulate all upcoming matchups, rank favorites by model win probability, export CSV.
- **Matchup simulator:** Choose teams, spread, total, simulations, home-field assumptions; inspect score distributions and download every simulation.
- **Historical backtest:** Walk-forward predictions with only prior-date games; accuracy, Brier score, and margin mean absolute error.
- **Data & methodology:** See team input averages and limitations.

## Important limitations
This is a scoring-based prototype, **not** a validated professional betting model. It does not automatically pull market odds, injuries, weather, EPA, quarterback changes, or strength-of-schedule-adjusted efficiency. It is not live play-by-play, and 100,000 simulations do not ensure accurate probabilities. Backtests are descriptive and sensitive to settings. Feed access may fail or change without notice.

## Next upgrade path
Integrate licensed play-by-play EPA, verified injury/weather feeds, historical market closing lines, opponent-adjusted ratings, and out-of-sample calibration. A hosted web version can be deployed separately to Streamlit Community Cloud or a private server; this package is not already hosted.

## Publish as an iPhone-friendly website

This package is **not yet hosted**. To make a shareable website:

1. Create a GitHub repository and upload `app.py` and `requirements.txt` from the `nfl_sim` folder to the repository root.
2. Sign in to Streamlit Community Cloud at https://share.streamlit.io/ using GitHub.
3. Choose **Create app**, select your repository and branch, and set the main file path to `app.py`. Deploy.
4. Open the resulting `*.streamlit.app` address in iPhone Safari. Use Safari Share → Add to Home Screen for an app-like shortcut.

The web host, rather than your iPhone, runs the Python simulation. Feed access and availability depend on the external provider and hosting service. No live sportsbook odds, injury feeds, or EPA are included.
