### 🌸 Iris Flower Classification using Deep Learning

A complete end-to-end deep learning project for classifying Iris flower species using a Neural Network built with TensorFlow/Keras, deployed with Gradio, and tracked using MLflow.

### 🚀 Project Overview

This project predicts the species of an Iris flower (Setosa, Versicolor, or Virginica) based on its sepal and petal dimensions.
The workflow covers:

### Data preprocessing

Exploratory data analysis (EDA)

Model training with TensorFlow

Evaluation using classification metrics

Deployment with Gradio

Model tracking using MLflow

### 🧠 Model Architecture

The neural network used in this project has the following structure:

Layer Type	Neurons	Activation	Dropout
Input	8	ReLU	0.2
Hidden	6	ReLU	-
Output	3	Softmax	-

Optimizer: Adam
Loss Function: Categorical Crossentropy
Metric: Accuracy

### 📊 Dataset

Dataset Name: Iris Dataset
Source: UCI Machine Learning Repository

Features:

Sepal Length (cm)

Sepal Width (cm)

Petal Length (cm)

Petal Width (cm)

Target Classes:

Iris-setosa

Iris-versicolor

Iris-virginica

### 🔍 Exploratory Data Analysis

The notebook includes:

Missing value and duplicate checks

Distribution plots with skewness

Correlation heatmap

Outlier detection using IQR

Boxplots and histograms


### 💻 Deployment with Gradio

A Gradio interface was created to make real-time predictions using the trained model.

▶️ Launch App Locally
pip install gradio tensorflow scikit-learn
python app.py

Then, open the local Gradio link in your browser to interact with the model UI.


🧩 Tech Stack
Category	Tools Used
Language	Python
Libraries	TensorFlow, Keras, Scikit-learn, Pandas, NumPy, Matplotlib, Seaborn
Deployment	Gradio
Experiment Tracking	MLflow
🧾 Project Structure
├── iris.ipynb           # Jupyter notebook with analysis and model
├── app.py               # Gradio app for deployment
├── requirements.txt     # Dependencies list
├── README.md            # Project documentation
└── saved_model/         # Trained model files

### 📦 Installation

Clone this repository:

git clone https://github.com/your-username/iris-flower-classification.git
cd iris-flower-classification


### Install dependencies:

pip install -r requirements.txt


Run the Jupyter notebook:

jupyter notebook iris.ipynb

## 🌐 Live Demo (Gradio)

You can access the live version of this project here:
👉 https://huggingface.co/spaces/nandha-01/IRIS-ANN

✨ Results

Achieved ~96% accuracy on test data.

Deployed successfully using Gradio interface.

Model performance tracked using MLflow for reproducibility.
