# Poultry Disease Classification

A specialized Deep Learning application capable of classifying poultry diseases from fecal images with high accuracy. This project implements a complete **MLOps pipeline** including data ingestion, model training (VGG16), evaluation, and a Flask web interface, all containerized with Docker.

## Features
- **Data Pipeline:** Automated ingestion and validation of image datasets.
- **Deep Learning:** Uses Transfer Learning (**VGG16**) for robust image classification.
- **Web Interface:** User-friendly Flask application for real-time predictions.
- **Containerization:** Fully Dockerized application for consistent deployment.
- **CI/CD Ready:** Structured for automated training and deployment workflows.

## Tech Stack
- **Language:** Python 3.8
- **Framework:** TensorFlow / Keras
- **Backend:** Flask
- **Container:** Docker
- **Tools:** Pandas, Numpy, Matplotlib

## Project Structure
```text
├── artifacts/          # Stores data, models, and training logs
├── config/             # Configuration files
├── src/                # Source code for the pipeline
├── templates/          # HTML files for the web app
├── app.py              # Flask application entry point
├── Dockerfile          # Blueprint for the container
├── requirements.txt    # Python dependencies
└── main.py             # Pipeline orchestrator



How to Run Locally
    Method 1: Using Docker (Recommended)
    This ensures the app runs in an isolated environment exactly as intended.

Build the Image:

Bash
docker build -t poultry-app .
Run the Container:

Bash
docker run -p 8080:8080 poultry-app
Access the App: Open your browser and go tohttp://localhost:8080

    Method 2: Manual Installation
    Clone the repository:

Bash
git clone [https://github.com/rajatsoni136/Poultry-Disease-Classification.git](https://github.com/rajatsoni136/Poultry-Disease-Classification.git)

Install dependencies:

Bash
pip install -r requirements.txt
Run the application:

Bash
python app.py
Model Training
If you want to retrain the model from scratch:

Update params.yaml(e.g., change EPOCHS).

Run the training command:

Bash
python stage_03_training.py
📊 Results
The model currently distinguishes between Healthy and Coccidiosis classes.

Base Model: VGG16 (ImageNet weights)

Accuracy: ~90% (after 5 epochs)