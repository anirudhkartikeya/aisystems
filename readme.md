# PyBullet Navigation: Playground

Data collection work for a simulated mobile robot navigation project that compares a reactive approach with a predictive one (pedestrian trajectory prediction). The PyBullet simulator is not built yet, so this repo uses a real pedestrian trajectory dataset as a stand-in.

## Dataset
- Name: ETH/UCY pedestrian trajectories (ETH, Hotel, Univ, Zara1, Zara2), preprocessed
- Downloaded from: https://github.com/InhwanBae/NPSN (folder `dataset/`)
- Version: commit (Version: commit 9d2fca3 (full hash 9d2fca316feb6b1235ace0184a0a4d8838dd5436))
- Retrieved: Oct 5, 2026
- Original sources: ETH (BIWI Walking Pedestrians): https://data.vision.ee.ethz.ch/cvl/aem/ewap_dataset_full.tgz. UCY: https://graphics.cs.ucy.ac.cy/research/downloads/crowd-data
- Storage: Google Drive, folder pybullet-nav-data/eth_ucy/raw/dataset (about 19 MB). The data is not stored in this repo.
- Personal data: none (anonymous x/y coordinates only)

## License
- Repo code license: MIT License (Copyright (c) 2022 Inhwan Bae), covering the NPSN code.
- Data terms: no explicit data license found in the repo. Redistribution terms are unconfirmed, so the data is kept out of this repo.

## How to run
1. Open playground.ipynb in Google Colab.
2. Mount Google Drive when prompted (the data folder must be in your Drive at the path above).
3. Run all cells from top to bottom.

Python version: (Python 3.13.16)

## Data quality findings

The table has 8 recordings (the `scene` column), and 2,205 pedestrian tracks in total, where a track is one pedestrian within one recording. There are no missing values in any column and no duplicate (scene, frame, pedestrian) rows, so the files are clean at that level.

The main problem is track length. A track has a median of 26 points (mean 33.8), but the shortest has only 2 and the longest has 584. 784 of the 2,205 tracks (about 36%) have fewer than 20 points. Predicting a path needs observed points plus future points, and 20 is a threshold I chose from the common 8-observed plus 12-predicted setup used in trajectory prediction benchmarks. So about a third of the tracks may be too short to use as prediction samples. I have not decided how to handle them yet. Options are filtering them out or only using tracks above a minimum length.

The coordinates are not in one shared frame. The ETH recording ranges from x -7.69 to 14.42 and y -3.17 to 13.21, and the Hotel recording from x -3.25 to 4.35 and y -10.31 to 4.31. The other six recordings (the UCY ones) all sit in roughly the same range, x about -0.5 to 15.6 and y about -0.4 to 14.0. The values look like meters, but I have not confirmed the units. Combining recordings would need a per-recording check or normalization.

Unsolved problems and gaps:
- The data has only pedestrians and no robot, so it cannot show how people react to a robot.
- The recordings are outdoor or campus scenes, which may not match the PyBullet scenarios.
- The data terms are unconfirmed (no explicit data license found).
- The repo also holds train/val/test splits of the same recordings, so I used only `all_data` to avoid duplicates.