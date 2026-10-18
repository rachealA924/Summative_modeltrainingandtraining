# Cassava Leaf Disease Classification: Classical ML vs Deep Learning

Summative assignment, Model Training and Evaluation. 
Author: Racheal Resty Akello

## Links
- Google Drive (raw data zip, best models, results): <https://drive.google.com/drive/folders/1pFZDtp62AhpwDWWG0D7KYr2bGlpKJadu?usp=drive_link>
- Presentation video: <>
- Report: submitted as PDF

## Problem
Cassava is a staple crop for millions of smallholder farmers in sub-Saharan Africa, and diseases such as mosaic and brown streak can destroy yields when they are spotted late. This project classifies a single cassava leaf photo into one of five classes: bacterial blight (CBB), brown streak (CBSD), green mottle (CGM), mosaic (CMD) or healthy. It compares classical machine learning with deep learning on 21,397 labelled images, using macro-F1 because the classes are strongly imbalanced.

## Dataset
Kaggle "Cassava Leaf Disease Classification" (21,397 labelled leaf images, 5 classes: CBB, CBSD, CGM, CMD, Healthy). Class imbalance: CMD 61.5%, CBB 5.1%. The raw data is **not** in this repo. Download it from Kaggle (accept the competition rules first) or use the Drive folder above.

## Approach
- Classical ML (scikit-learn): Logistic Regression, Random Forest, Gradient Boosting on frozen MobileNetV2 features (E1–E6)
- Deep learning (Keras, tf.data): MobileNetV2 transfer learning with a trained head, fine-tuning, augmentation and class weights (E7–E11)
- Evaluation: accuracy, macro-F1, macro-AUC, per-class precision/recall, confusion matrices, ROC curves, learning curves
- Experiments tracked with Weights & Biases (offline mode)

## Results (test set)
| Model | Accuracy | Macro-F1 | Macro-AUC |
|---|---|---|---|
| E4 Logistic Regression (classical) | 0.732 | 0.549 | 0.881 |
| E11 MobileNetV2 + class weights (deep) | 0.728 | 0.571 | 0.896 |

## Reproduce
1. Open `notebooks/Summative_RachealRestyAkello.ipynb` in Google Colab (GPU runtime: T4).
2. Add your Kaggle API token as a Colab secret named `KAGGLE_API_TOKEN`.
3. Run all cells top to bottom. Seed 42 is set; small run-to-run differences on GPU are normal.
4. `pip install -r requirements.txt` for local use.

## Notes
The classical models use features from a pretrained MobileNetV2 (ImageNet weights). Data is from Kaggle and shared via Drive for assessment only.
