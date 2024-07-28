# Tuberculosis-Prediction-Based-on-Chest-X-Rays-with-tensorflow
The project demonstrated the effective use of transfer learning with DenseNet121 for TB detection from chest X-rays, achieving high accuracy and robustness.

# To successfully run the tuberculosis prediction project using chest X-rays dataset with TensorFlow, ensure your system meets the following requirements:

# Hardware
CPU: Intel Core i5 or equivalent (minimum)
GPU: NVIDIA GPU with CUDA support (recommended for faster training)
RAM: 16 GB (minimum)
Storage: 50 GB free space (for dataset, libraries, and logs)

# Software

Operating System:
Windows 10 or later
Ubuntu 18.04 or later
macOS Catalina or later
Python Version: 3.6 or later


# Library Requirements
Ensure the following Python libraries are installed:

TensorFlow:
Keras (already included with TensorFlow):
NumPy:
Matplotlib:
scikit-learn:
imblearn:
Pandas:
ImageDataGenerator (part of Keras):

# Setup Instructions

Clone the Project Repository:
Set Up Virtual Environment (optional but recommended):
Install Required Libraries:


# Dataset Preparation
## Download the Dataset:

Download the chest X-ray images from the specified sources (e.g., Tuberculosis (TB) Chest X-ray Database, Montgomery County X-ray Set, TBX11K).
Ensure the dataset is organized into folders representing each class (Normal and Tuberculosis).
Organize the Dataset:

Place the dataset in a directory structure as follows:
css
Copy code
dataset/
├── Normal/
│   ├── image1.png
│   ├── image2.png
│   └── ...
└── Tuberculosis/
    ├── image1.png
    ├── image2.png
    └── ...
    
# Running the Project
## Run the Baseline Model:

Navigate to the notebooks directory and open Base_model_KNN_combinedataset.ipynb in JupyterLab or Colab.
Execute the cells to run the baseline KNN model.

## Run the Final Model:

Open Final_beforefinetuning.ipynb in JupyterLab or Colab.
Execute the cells to run the transfer learning model with DenseNet121.


## Run the Fine-Tuning Model:

Open FullandFinal.ipynb in JupyterLab or Colab.
Execute the cells to run the fine-tuning model with DenseNet121 by unfreezing the last 5 layers.


# Additional Tools
JupyterLab or,
Google Colab: For those who prefer to use cloud-based notebooks.

# Notes
CUDA and cuDNN: If using an NVIDIA GPU, ensure CUDA and cuDNN are properly installed to leverage GPU acceleration with TensorFlow.
Environment Variables: Configure environment variables for CUDA if necessary.
By following these instructions, you should have a fully operational environment ready to train and evaluate the tuberculosis prediction models using chest X-rays with TensorFlow.

## Getting Started

To get started with this project, reading of `Project-Report.pdf` (aout 15 page in maximum) is highly recommendable before approaching the code.

