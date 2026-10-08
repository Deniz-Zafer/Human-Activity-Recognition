# Human Activity Recognition with Smartphone Sensors

Classification of six human activities (walking, walking upstairs, walking downstairs, sitting, standing, laying) from smartphone accelerometer and gyroscope data using the **UCI HAR dataset**. Introduction to Data Science course project.

## Models

| Model | Input | Accuracy |
|---|---|---|
| Support Vector Machine | 561 hand-crafted features (standardized) | ~98.3 % |
| Random Forest (300 trees) | 561 hand-crafted features | ~98.0 % |
| 1D-CNN (TensorFlow/Keras) | Raw inertial signals | ~92.4 % |

The notebook also includes confusion matrices, per-class precision / recall / F1 reports, Random Forest feature-importance analysis and a model comparison chart.

## Running
Open `Human_Activity_Recognition_Analysis.ipynb` in Google Colab or Jupyter. The dataset is downloaded via KaggleHub (or set `DATASET_DIR` to a local copy).

**Authors:** Gülsüm Durmaz, Deniz Zafer
**Tools:** Python, pandas, scikit-learn, TensorFlow/Keras, matplotlib, seaborn
