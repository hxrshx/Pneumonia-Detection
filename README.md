# Computer Vision - Portfolio Exam 2

# Assignment 1: Explainable AI for Medical Imaging - Interpreting Chest X-Ray Pneumonia Predictions

**Submitted by:**

* Riya Biju - 10000742
* Harsha Sathish - 10001000
* Harshith Babu Prakash Babu - 10001191

---

## Project Overview

This project implements Explainable AI (XAI) techniques for pneumonia detection in chest X-ray images. The system uses a ResNet-18 based classifier with two interpretability methods - Grad-CAM and LIME - to provide visual explanations for model predictions, enhancing trust and transparency in medical AI diagnostics.

**Key Features:**
* Binary classification of chest X-rays (Normal vs Pneumonia)
* ResNet-18 architecture fine-tuned for medical imaging
* Grad-CAM visualization for class-discriminative localization
* LIME explanations for interpretable feature importance
* Comparative analysis of XAI methods on clinical data

---

## Prerequisites

**Dataset:**

Download the Chest X-Ray Pneumonia Dataset from Kaggle:
* Dataset URL: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
* Extract the dataset to your working directory

**Python Environment:**

Create a virtual environment and install dependencies:

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate

# Install required packages
pip install -r requirements.txt
```

---

## Repository Structure

```
CV_Portfolio2_Riya_Harsha_Harshith/
│
├── Assignment-1_PneumoniaDetection/           # Assignment 1 folder
│   ├── README.md                               # This file
│   ├── Presentation.pdf                        # The presentation
│   ├── Portfolio2_CV.ipynb                     # Main Jupyter notebook
│   ├── requirements.txt                        # Python dependencies
│   │
│   ├── best_model.pth                          # Trained ResNet-18 model (best epoch)
│   │
│   ├── lime_vs_grad-cam_normal.png             # XAI comparison on normal X-ray
│   ├── lime_vs_grad-cam_pneumonia.png          # XAI comparison on pneumonia X-ray
│   ├── evaluation_normal.png                   # Evaluation metrics for normal cases
│   ├── evaluation_pneumonia.png                # Evaluation metrics for pneumonia cases
│   └── accuracy_and_loss.png                   # Training curves
│
└── Assignment-2_EdgeAI/                        # Assignment 2 folder
```

---

## Results Overview

### Output Files:

1. **lime_vs_grad-cam_normal.png**
   * Comparative visualization of LIME and Grad-CAM on a normal chest X-ray
   * Shows agreement/disagreement between explanation methods

2. **lime_vs_grad-cam_pneumonia.png**
   * Comparative visualization on pneumonia-positive X-ray
   * Highlights regions indicating infection

3. **evaluation_normal.png**
   * Performance metrics for normal X-ray classification
   * Includes precision, recall, and confidence distributions

4. **evaluation_pneumonia.png**
   * Performance metrics for pneumonia detection
   * Sensitivity analysis for clinical deployment

5. **accuracy_and_loss.png**
   * Training and validation curves over epochs
   * Shows convergence and potential overfitting patterns

---

## Dependencies

Key libraries required (see `requirements.txt` for complete list):
* PyTorch (deep learning framework)
* torchvision (pre-trained models and transforms)
* numpy (numerical operations)
* matplotlib (visualization)
* opencv-python (image processing)
* lime (LIME explanations)
* Pillow (image loading)
* scikit-learn (evaluation metrics)

---

## Acknowledgments

* **Instructor:** Prof. Dr. Dominik Seuß - Computer Vision Course, THWS
* **Dataset:** Chest X-Ray Pneumonia Dataset by Paul Mooney (Kaggle)
