Where to find this assignment (First Model Exploration)

The repository contains the whole project (simulator, brains, planner). For this assignment, only these files matter:

training_experiments.ipynb: the assignment notebook (all six steps, with outputs).
results/experiments.csv: the experiment table (one row per run).
results/summary_note.txt: the short summary note.
results/data_overview.png, results/run1_curves.png, results/run1_worst_cases.png: the plots.
results/lstm_best.pt: the trained LSTM model.
make_data.py: the script that generated the dataset (python make_data.py --episodes 500).

The dataset itself (trajectories/, about 125 MB) is not in the repository because of its size. It can be regenerated with the command aboveh the PyBullet scenarios.
- The data terms are unconfirmed (no explicit data license found).
- The repo also holds train/val/test splits of the same recordings, so I used only `all_data` to avoid duplicates.
