# Dataset setup

This project uses the labelled Titanic training dataset from Kaggle's **Titanic - Machine Learning from Disaster** competition.

1. Open https://www.kaggle.com/competitions/titanic/data and sign in if asked.
2. Accept the competition rules if Kaggle requests it.
3. Download `train.csv` from the Data tab.
4. Place the file here so the path is `data/train.csv`.

`train.csv` is deliberately not committed to this repository. Kaggle's competition data is subject to its competition rules; retrieve it from the original source. The training file includes the `Survived` label. Kaggle's `test.csv` does not include labels, so it cannot be used to calculate validation accuracy.
