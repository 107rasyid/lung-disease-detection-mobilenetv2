# Lung Disease Detection Using MobileNetV2

This project provides a web application built using **Streamlit** for classifying lung diseases using **MobileNetV2**. The model is designed to classify chest X-ray images into various categories related to lung diseases.

## Live Demo

You can access the live demo of the application here:

[Live Demo - Lung Disease Detection](https://lung-disease-detection-mobilenetv2-c9gt8f74swbvyvptafwyg9.streamlit.app/)

## Features

- Upload chest X-ray images.
- The model classifies the image into one of the following categories:
  - **NORMAL** (Normal lung)
  - **TUBERCULOSIS** (Tuberculosis)
  - **PNEUMONIA** (Pneumonia)
  - **COVID-19** (COVID-19 infection)
  
- View prediction results along with class probabilities.

## Requirements

Before running the application locally, ensure that the following dependencies are installed:

- Python 3.x
- TensorFlow
- Streamlit
- OpenCV

### Installation

You can install the necessary dependencies using `pip` by creating a virtual environment and running:

```bash
pip install -r requirements.txt
```

### How to Run Locally

Clone the repository to your local machine:
```bash
git clone https://github.com/yourusername/lung-disease-detection-mobilenetv2.git
cd lung-disease-detection-mobilenetv2
```

Run the Streamlit app:
```bash
streamlit run app.py
```
