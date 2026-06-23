# MLB DFS Stack Command Center

A Streamlit app that ranks MLB DFS stacks by using:

1. Scoring % Sheet as the slate master
2. DK Salaries for game/home park mapping and player pool
3. Matchups Master for trend/matchup context
4. Park Factors for run/HR environment

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Upload order

- Scoring % Sheet
- Matchups Master Sheet
- Park Factors Sheet
- DK Salaries Sheet

The app only ranks teams from the scoring sheet.
