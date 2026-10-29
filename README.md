# NFL Prediction Command Center — Game + Player Props

## Deploy to Streamlit Community Cloud
Upload `app.py` and `requirements.txt` as **separate files** at the ROOT of your GitHub repository. Do not upload only this ZIP.

In Streamlit deployment use your repo, your actual branch (typically `main`), and main file path `app.py`.

## Player props
At the bottom of the app open **Player Prop Lab**, click **Load player game logs**, then choose individual simulations or rankings. Rankings use median historical stats as reference lines, **not** live sportsbook prices. ESPN public boxscore feeds are unofficial and can fail or change.

## Caveats
Team simulator uses scoring statistics. Player props use empirical recent game logs, not opponent-adjusted forecasts; predictions are exploratory and uncalibrated. Live odds, injuries and EPA are not integrated. The app needs network access. For performance, start with 2-4 weeks of player data.
