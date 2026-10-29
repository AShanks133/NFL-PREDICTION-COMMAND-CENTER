# NFL Prediction Command Center — Game Theater Edition

Includes weekly team Monte Carlo forecasts, matchup simulator, player prop lab, and a **fictional animated play-by-play game theater**.

## Deploy from iPhone

1. Extract this ZIP in iOS Files.
2. Upload **app.py** and **requirements.txt** separately to the root of your GitHub repository (replace old files). Do **not** upload only the ZIP.
3. In Streamlit Community Cloud, choose the repo and correct branch, with main file `app.py`.
4. Deploy, or reboot your existing app if needed.

## Theater controls

Scroll to **Live Game Theater**. Choose teams, Kickoff / Reset, and then Next play, or select a non-manual speed and tap Play. Use Finish game to run through to the final score. Field position and a synthetic team box score update on each play.

## Caveats

- This is a **fictional play-by-play animation**, not live NFL tracking, footage, or Madden-quality animated athletes.
- Player props are a separate empirical historical-bootstrap tool and **do not update from the fictional game theater**.
- The play generator is an illustrative approximation, not a calibrated play-by-play NFL model. The original team forecast engine is separate and does not derive its win probabilities from the theater.
- Data loading requires access to ESPN's unofficial public endpoints; availability may vary.
