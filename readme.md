# AI Systems coursework (University of Florida, Fall 2026)

Coursework repository of Pisipati Kartikeya Anirudh (M.S. AI Systems).

## Main project: SEE YOU COMING

**[robot_nav_project/](robot_nav_project/)** is a simulated self-driving car that predicts where pedestrians are
going. It uses a PyBullet town simulator, three swappable prediction brains (reactive, constant velocity, LSTM), one
shared planner, a fine-tuned YOLO11 person detector, a reproducible evaluation and trustworthiness checks
(LIME, input importance, stress tests). On held-out seeds, prediction cut runs that touch a person from 70% to 0-5%.

See the **[project README](robot_nav_project/README.md)** for the results and the
**[run guide](robot_nav_project/HOW_TO_RUN.md)** for all the commands.

## Assignments in this repository

### Data collection playground (`playground.ipynb`)
Written before the simulator existed, it explores a real pedestrian dataset as a stand-in: ETH/UCY trajectories,
preprocessed, from [InhwanBae/NPSN](https://github.com/InhwanBae/NPSN) at commit 9d2fca3. The data is kept out of
this repo because its redistribution terms are unconfirmed. The notebook runs in Google Colab with the data in
Google Drive (`pybullet-nav-data/eth_ucy/raw/dataset`). Only `all_data` was used, to avoid the duplicate train/val/test copies.

### First Model Exploration (in `robot_nav_project/`)
- `training_experiments.ipynb`: the assignment notebook (all six steps, with outputs).
- `results/experiments.csv`: the experiment table (one row per run).
- `results/summary_note.txt`: the short summary note.
- `results/data_overview.png`, `results/run1_curves.png`, `results/run1_worst_cases.png`: the plots.
- `results/lstm_best.pt`: the trained LSTM model.
- `make_data.py`: the script that generated the dataset (`python make_data.py --episodes 500`).

The dataset itself (`trajectories/`, about 125 MB) is not in the repository because of its size. It can be
regenerated with the command above.

> Note added later: the LSTM has since been retrained (LSTM v2 with context) and now drives the car as the third
> brain. See the project README. The text below describes the assignment as submitted.

This is a first training sample; the LSTM is not yet integrated into the car’s driving loop (the car still drives with the constant-velocity and reactive brains, and plugging the LSTM in is the next step). I explored the prediction stage first because the planner depends on it; YOLO detection comes next. The selected candidate is a one-layer LSTM (hidden 64) that predicts a pedestrian’s next 8 positions (3.2 s) from their recent track. Initial result on 56,021 validation examples: ADE 0.566 m and FDE 1.205 m, slightly better than constant velocity (0.583 m / 1.236 m) and far better than stay-put (1.599 m / 2.793 m). Observed failure: the largest errors come from people who change what they are doing (stopping at the kerb, starting to cross): 54% of the worst 5% of cases, against 31% of all examples. Inference takes 0.22 ms on CPU for 30 people, well within the 100 ms target. Targeted change: a longer history (12 points instead of 8) made no clear difference (ADE 0.567 m, FDE 1.205 m). The test split was not used.

## Experiment results (First Model Exploration)

> First training sample only. The LSTM is not yet integrated into the car's driving loop.

| Run | Configuration | Change | ADE (m) | FDE (m) | Val examples | Train time (s) | Inference, 30 people (ms) |
|---|---|---|---|---|---|---|---|
| Stay-put (reactive brain) | no learning | – | 1.599 | 2.793 | 56,021 | 0 | – |
| Constant velocity (predictive brain) | no learning | – | 0.583 | 1.236 | 56,021 | 0 | – |
| LSTM run 1 | 1 layer, hidden 64, history 8 pts, lr 0.001, batch 512, 5 epochs, seed 0 | first configuration | 0.566 | 1.205 | 56,021 | 4.3 | 0.22 |
| LSTM run 2 | 1 layer, hidden 64, history 12 pts, lr 0.001, batch 512, 5 epochs, seed 0 | longer history (4.4 s instead of 2.8 s) | 0.567 | 1.205 | 56,021 | 4.9 | 0.31 |

- **ADE:** average distance (m) between predicted and real positions over the next 3.2 s. **FDE:** the same, at 3.2 s ahead. Lower is better.
- Evaluated on the validation split (whole simulator runs, none of them used for training). The test split was not used.

