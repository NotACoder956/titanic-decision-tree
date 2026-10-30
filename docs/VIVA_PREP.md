# Reviewer / viva preparation

These are short answers you can explain in your own words. Do not memorise them blindly; understand the reason behind each choice.

## 1. What problem does this project solve?

It learns from labelled Titanic passenger records and predicts the binary `Survived` label (0 = did not survive, 1 = survived) from `Pclass`, `Sex`, `Age`, and `Fare`. This is supervised binary classification.

## 2. Why did you choose a Decision Tree?

A Decision Tree is suitable for a beginner project because it supports classification and is interpretable. It makes threshold-based decisions that can be visualised. The maximum depth is restricted so the model does not grow into a very large tree.

## 3. What is the target and what are the features?

`Survived` is the target (what the model predicts). `Pclass`, `Sex`, `Age`, and `Fare` are input features (the information used to make the prediction).

## 4. Why do you split the dataset?

The model learns from the training rows. The validation rows are held aside to estimate how it performs on passengers it did not use for fitting. Testing on training rows alone would give an overly optimistic result.

## 5. What does `stratify=y` do?

It preserves roughly the same proportion of survived and non-survived passengers in both splits. That makes the holdout comparison more representative of the original target distribution.

## 6. Why are missing values handled?

Many machine-learning estimators cannot accept missing values directly. The dataset has missing `Age` values. Numeric inputs use median imputation because it is robust to extreme values; the categorical `Sex` field uses the most frequent category.

## 7. Why encode `Sex`?

The raw categories `female` and `male` are text. One-hot encoding represents them as numeric indicator columns so the classifier can use them without inventing an ordinal relationship between the categories.

## 8. Why use a `Pipeline` and `ColumnTransformer`?

They put preprocessing and classification into one reusable object. Imputation and encoding are fitted on training rows only and then applied to validation rows, which helps prevent data leakage and keeps the same transformations for future predictions.

## 9. What does 79.3% accuracy mean here?

On this particular 179-row validation split, about 79.3% of predictions matched the known `Survived` labels. It is not a promise about every passenger or a score on Kaggle's unlabelled test.csv.

## 10. Why compare against a majority-class baseline?

The dataset has more non-survivors than survivors. Predicting the most common class for everyone gives 61.5% accuracy on this split. The model's 79.3% accuracy is about 17.9 percentage points higher, so the model is learning patterns beyond always choosing the majority class.

## 11. What do precision, recall, and F1 mean?

- **Precision for survived:** of the passengers the model predicted would survive, the fraction who actually survived.
- **Recall for survived:** of the passengers who actually survived, the fraction the model correctly identified.
- **F1-score:** the harmonic mean of precision and recall, summarising the balance between them.

Recall matters here because the model misses some actual survivors; the confusion matrix makes those false negatives visible.

## 12. Which feature is most important?

In the included run, `Sex` is the most important input (about 64.4% aggregated importance), followed by `Pclass` (about 21.9%). The importance of the encoded female/male columns is combined into one `Sex` score for the graph.

## 13. Does feature importance prove that sex caused survival?

No. Importance explains how this fitted decision tree used features to reduce impurity. It is model-specific and does not establish causation. The historical data may also reflect social and evacuation patterns that are not represented by these four features.

## 14. What does the confusion matrix tell you?

In the validation set, it contains 97 true negatives, 13 false positives, 24 false negatives, and 45 true positives. It reveals the types of errors, not just the overall percentage correct.

## 15. What would you improve next?

I would compare the tree with a simple baseline and other models, use cross-validation for more stable model selection, inspect class-wise errors, and try additional features only when justified. I would keep a final untouched test set for the final estimate.

## 16. How do you run the project?

Download Kaggle's labelled `train.csv` into `data/train.csv`, install `requirements.txt`, then run `PYTHONPATH=src python -m titanic_tree.train`. The script prints evaluation metrics and saves charts, CSV summaries, JSON metrics, and a fitted pipeline. Tests run with `PYTHONPATH=src pytest -q`.
