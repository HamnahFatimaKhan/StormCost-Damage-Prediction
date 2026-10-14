# StormCost: Property Damage Prediction

A machine learning project for estimating property damage from severe weather events by combining structured storm records with information from event narratives.

**Competition result:** 15th place  
**Team:** Whethered_Minds  
**Team size:** 2  
**Competition:** StormCost

## Project Overview

Predicting storm-related property damage is difficult for two main reasons: many records report no property damage, while the recorded losses for damaging events vary widely. The data also spans different historical periods, which can introduce changes in the distribution of events and damage amounts.

Our approach treats the task as two connected problems:

1. Estimate whether an event is likely to have positive property damage.
2. Estimate the damage amount when positive damage is predicted.

The final solution combines a LightGBM-based model with a fine-tuned MiniLM model. Predictions are blended and calibrated using validation results.

## Approach at a Glance

```text
Storm records and narratives
             |
             v
   Feature preparation
             |
       +-----+------+
       |            |
       v            v
Structured data   Text narratives
       |            |
       |       MiniLM embeddings
       |            |
       |           PCA
       +-----+------+
             |
             v
      Combined features
             |
       +-----+------+
       |            |
       v            v
    LightGBM     Fine-tuned
     hurdle        MiniLM
      model         model
       |            |
       +-----+------+
             |
             v
  Validation-based blending
             |
             v
  Historical-period calibration
             |
             v
  Property damage predictions
```

## 1. Preparing the Features

The modeling pipeline uses both tabular fields and narrative text.

### Structured features

Features include information such as:

- Event year and month
- Latitude, longitude, and state
- Event type and magnitude information
- Tornado-related measurements where available
- Combined injury and death counts
- Event and episode narrative lengths
- Indicators for missing episode narratives

Categorical fields are prepared for LightGBM, while numerical fields are incorporated into the combined feature matrix.

### Narrative features

We used `sentence-transformers/all-MiniLM-L6-v2` to represent the event narrative and the wider episode narrative as semantic embeddings.

- Event narrative embeddings were reduced to 32 principal components.
- Episode narrative embeddings were reduced to 24 principal components.
- Repeated text entries were encoded once and then mapped back to their records.

Using both narrative levels helps represent details about an individual storm as well as the broader weather episode.

## 2. Modeling Damage Occurrence and Magnitude

### LightGBM hurdle model

The LightGBM component has two parts.

**Damage classifier:** predicts the probability that recorded property damage is greater than zero.

**Magnitude regressor:** trains on positive-damage records and predicts the transformed target:

```python
np.log1p(damage)
```

The logarithmic transformation helps manage the highly skewed distribution of loss amounts. Predictions are converted back to the original scale using `expm1`, followed by validation-based magnitude scaling.

The magnitude model also uses target-dependent sample weights so that larger losses receive additional influence during training.

### Fine-tuned MiniLM

We also fine-tuned MiniLM with PyTorch and Hugging Face Transformers. The neural model uses event context and episode narrative text to learn task-specific representations.

Its prediction head produces two outputs:

- A probability estimate for positive damage
- A transformed estimate of damage magnitude

The training objective combines a binary classification loss with a magnitude-regression loss for positive-damage examples. This allows the model to learn both parts of the problem through a shared text encoder.

## 3. Validation and Leakage Control

We used **three-fold GroupKFold validation**, grouping records by `episode_group`.

Records from the same episode were kept together within a fold to reduce the risk of related observations appearing in both training and validation data. The same fold assignments were used to align predictions across model components.

Out-of-fold predictions supported the selection of blending weights and calibration parameters. Neural-model training was time-limited, so the number of completed folds could depend on available runtime.

## 4. Blending and Historical Calibration

LightGBM and fine-tuned MiniLM contribute different strengths:

- LightGBM models relationships among structured fields and compressed text features.
- MiniLM learns task-specific context from event and episode narratives.

We evaluated candidate blends using validation predictions rather than assuming that the more complex model would always perform better. The LightGBM-only result was retained as a fallback, and the blended approach was selected only when it cleared the configured validation-improvement margin.

The final decision layer used separate probability thresholds for older and more recent storms, with the historical split defined as:

```python
old = year <= 2005
```

A magnitude multiplier was also selected during validation. The final prediction is set to zero when the occurrence probability does not pass the applicable threshold; otherwise, the calibrated magnitude estimate is used. Predictions are constrained to be nonnegative.

## 5. Submission Checks

Before generating the competition submission, the pipeline checks that:

- The prediction count matches the test set.
- Sample identifiers are unique.
- Predictions are finite and nonnegative.
- The output contains the required columns: `sample_id` and `predicted_damage`.

## Results

Our team achieved **15th place** in the StormCost competition.

The final approach brought together structured feature engineering, narrative embeddings, a two-part damage model, a fine-tuned language model, and validation-driven calibration.

## Technology Stack

- **Language:** Python
- **Tabular machine learning:** LightGBM, scikit-learn
- **NLP and embeddings:** Sentence Transformers, MiniLM
- **Neural modeling:** PyTorch, Hugging Face Transformers
- **Data processing:** NumPy, Pandas
- **Dimensionality reduction:** PCA
- **Validation:** GroupKFold
- **Tools:** Jupyter Notebook, Git, GitHub

## Challenges and Takeaways

- Zero-damage records and high-loss records require different modeling treatment.
- Narrative text can add context that is not fully represented by structured variables.
- Group-based validation is important when records share a broader event episode.
- Historical differences may call for calibration beyond a single global threshold.
- More complex models should be kept only when validation supports their contribution.

## Possible Next Steps

Potential extensions include testing alternative text encoders, exploring time-based validation, improving probability calibration, and evaluating additional approaches for unusually large losses.

## Authors

- **Hamnah Fatima Khan**  
- **Mehak Gul** 
## Project Reference

- [StormCost Kaggle competition write-up](https://www.kaggle.com/competitions/storm-cost/writeups/15th-Place-Solution)

## License and Data

This repository is intended for educational and competition-related use. Review the original competition rules and dataset terms before sharing any competition data or derived datasets. Add a repository license only if you have chosen one for this project.
