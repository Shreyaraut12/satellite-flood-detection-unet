# 🛰️ Satellite Flood Detection Using U-Net

A deep learning application for **flood-region segmentation in satellite imagery** using a U-Net convolutional neural network, PyTorch, and Streamlit.

## 📌 Project Overview

Floods can cause significant damage to infrastructure, agriculture, and human settlements. Satellite imagery combined with deep learning can help identify potential flood-affected regions through pixel-level image segmentation.

This project implements a U-Net model to predict flood regions from satellite images and provides an interactive Streamlit interface for visualizing the predictions.

## ✨ Features

- Satellite image upload through a web interface
- U-Net-based binary image segmentation
- Configurable flood detection probability threshold
- Predicted binary flood mask
- Red overlay highlighting predicted flood regions
- Grayscale flood probability map
- Predicted flood-pixel percentage
- Downloadable prediction outputs

## 🧠 Model Architecture

The project uses **U-Net**, an encoder–decoder convolutional neural network designed for image segmentation.

- **Encoder:** Extracts spatial features at multiple resolutions.
- **Bottleneck:** Learns deeper feature representations.
- **Decoder:** Reconstructs spatial information using upsampling.
- **Skip connections:** Combine encoder and decoder features to preserve spatial detail.
- **Output:** A single-channel probability map for binary flood segmentation.

The model takes RGB satellite images resized to 256 × 256 pixels and predicts potential flood pixels.

## 🛠️ Technology Stack

- Python
- PyTorch
- Torchvision
- NumPy
- Pillow (PIL)
- Streamlit
- Jupyter Notebook

## 📂 Project Structure

```text
Satellite-Flood-Detection-UNet/
├── app.py
├── model.py
├── train.ipynb
├── requirements.txt
├── final_model_comparison.csv
├── result.png
├── .gitignore
└── README.md
```

The dataset and trained model checkpoints are not included in this repository.

## 📊 Experimental Results

Three U-Net configurations were evaluated on a held-out test set of 44 image–mask pairs using a probability threshold of 0.30.

| Metric | Baseline | Fine-tuned | Augmented |
|---|---:|---:|---:|
| Precision | 71.62% | **81.71%** | 73.43% |
| Recall | **90.08%** | 83.68% | 86.91% |
| Dice / F1 | 79.80% | **82.68%** | 79.60% |
| Global IoU | 66.38% | **70.48%** | 66.12% |
| Pixel Accuracy | 82.67% | **86.68%** | 83.08% |

The fine-tuned model achieved the highest Dice/F1 score and global IoU among the evaluated configurations. The baseline achieved the highest recall.

These results describe the evaluated test set and do not guarantee equivalent performance on unseen geographic regions or different satellite imagery.

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/Shreyaraut12/satellite-flood-detection-unet.git
cd satellite-flood-detection-unet
```

### 2. Create a virtual environment

**Windows PowerShell:**

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If your installed PyTorch version does not support your available hardware, install the appropriate version for your system using the official [PyTorch installation guide](https://pytorch.org/get-started/locally/).

## 🚀 Running the Application

Before running the app, obtain the trained model checkpoint and place it in the project directory. Make sure its filename matches the `MODEL_PATH` configured in `app.py`.

Start Streamlit:

```bash
python -m streamlit run app.py
```

Open the local URL printed in the terminal, usually:

```text
http://localhost:8501
```

Upload a satellite image, adjust the detection threshold if needed, and click **Detect Flood Regions** to view the predicted mask, overlay, probability map, and prediction summary.

## 📈 Evaluation Metrics

- **Precision:** Fraction of predicted flood pixels that are actual flood pixels.
- **Recall:** Fraction of ground-truth flood pixels detected by the model.
- **Dice/F1:** Measures overlap between predicted and ground-truth flood regions.
- **Intersection over Union (IoU):** Measures the intersection divided by the union of the predicted and ground-truth regions.
- **Pixel Accuracy:** Fraction of all pixels classified correctly.

## ⚠️ Limitations

- Predictions depend on the training data and the similarity of input imagery.
- Results may vary with satellite source, resolution, geography, and environmental conditions.
- The predicted flood-pixel percentage is a proportion of classified image pixels, not a direct measurement of flooded land area.
- The application is an experimental segmentation tool, not a substitute for official flood monitoring or emergency-response systems.
- Model weights must be obtained separately because they are not stored in this repository.

## 🔮 Future Improvements

- Evaluate on larger and geographically diverse datasets.
- Improve generalization through systematic augmentation and hyperparameter tuning.
- Add model checkpoint download instructions.
- Deploy the application for public access.
- Investigate geospatial area estimation using appropriate geographic metadata.

## 👩‍💻 Author

**Shreya Raut**

B.Tech — Artificial Intelligence and Machine Learning

GitHub: [@Shreyaraut12](https://github.com/Shreyaraut12)

---

*Developed as an academic deep learning project exploring satellite-image segmentation for potential flood-region detection.*
