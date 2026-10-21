# Cassava Leaf Disease Classification: Classical ML vs Deep Learning

Summative assignment, Model Training and Evaluation. Author: Racheal Resty Akello

## Links
- GitHub repository: https://github.com/rachealA924/Summative_modeltrainingandtraining
- Google Drive (dataset zip, best models, results): https://drive.google.com/drive/folders/1pFZDtp62AhpwDWWG0D7KYr2bGlpKJadu?usp=drive_link
- Presentation video: https://youtu.be/24nEcHpWN9U
- Report: submitted as PDF

## Problem
Cassava is a staple crop for millions of smallholder farmers in sub-Saharan Africa, and diseases such as mosaic and brown streak can destroy yields when they are spotted late. This project classifies a single cassava leaf photo into one of five classes: bacterial blight (CBB), brown streak (CBSD), green mottle (CGM), mosaic (CMD) or healthy. It compares classical machine learning with deep learning on 21,397 labelled images, using macro-F1 because the classes are strongly imbalanced.

## Dataset
Kaggle "Cassava Leaf Disease Classification" (21,397 labelled leaf images, 5 classes: CBB, CBSD, CGM, CMD, Healthy). Class imbalance: CMD 61.5%, CBB 5.1%. Stratified 70/15/15 split with seed 42 (14,977 / 3,210 / 3,210).

The raw data is **not** in this repo. Two ways to get it:
1. Download it from Kaggle (accept the competition rules first), which is what the notebook does.
2. Use `cassava_raw_labeled_dataset.zip` in the Drive folder above. It contains the raw labelled data (`train_images/`, `train.csv`, `label_num_to_disease_map.json`). Kaggle's unlabelled test images and TFRecord duplicates are left out because the notebook does not use them.

## Repository structure
- `notebooks/Summative_RachealRestyAkello.ipynb`: full pipeline with outputs (EDA, split, `tf.data`, 11 experiments, test evaluation, figures)
- `requirements.txt`: dependencies
- `README.md`

## Approach
- Classical ML (scikit-learn): Logistic Regression, Random Forest, Gradient Boosting on frozen MobileNetV2 features (E1–E6)
- Deep learning (Keras, tf.data): MobileNetV2 transfer learning with a trained head, fine-tuning, augmentation and class weights (E7–E11)
- Model selection: best validation macro-F1 in each family (E4 classical, E11 deep), then one evaluation on the held-out test set
- Evaluation: accuracy, macro-F1, macro-AUC, per-class precision/recall, confusion matrices, ROC curves, learning curves
- Experiments tracked with Weights & Biases (offline mode)

## Results (test set)
| Model | Accuracy | Macro-F1 | Macro-AUC |
|---|---|---|---|
| E4 Logistic Regression (classical) | 0.732 | 0.549 | 0.881 |
| E11 MobileNetV2 + class weights (deep) | 0.728 | 0.571 | 0.896 |

E11 raised CBB recall from 0.34 to 0.61 but lowered CBSD recall from 0.47 to 0.36. The full E1–E11 experiment table is in the report and notebook.

## Saved models and results (Google Drive)
- `models/best_classical_E4_logreg.joblib`
- `models/best_deep_E11.h5` (and `.keras`)
- `results/`: experiment and test result tables, data splits, figures, W&B offline logs

## Reproduce
1. Open `notebooks/Summative_RachealRestyAkello.ipynb` in Google Colab (GPU runtime: T4).
2. Add your Kaggle API token as a Colab secret named `KAGGLE_API_TOKEN` and accept the competition rules.
3. Run all cells top to bottom. Seed 42 is set; small run-to-run differences on GPU are normal.
4. `pip install -r requirements.txt` for local use.

Without Kaggle access: download `cassava_raw_labeled_dataset.zip` from Drive, upload it to Colab, run `!mkdir -p /content/data && unzip -q cassava_raw_labeled_dataset.zip -d /content/data`, and skip the Kaggle download cell.

## Limitations
One stratified split of one dataset, so there is no external validation; the classical models use pretrained deep features, so they are not independent of deep learning. This is a research comparison, not a field-ready diagnostic tool.

## Notes
The classical models use features from a pretrained MobileNetV2 (ImageNet weights). Data is from Kaggle and shared via Drive for assessment only.
