# Airbnb Price Prediction with AWS Deployment

## Table of Contents

- [Description](#description)
- [Tech Stack](#tech-stack)
- [Project Structure (Simplified / Key Files)](#project-structure-simplified--key-files)
- [Installation / Setup](#installation--setup)
- [Usage](#usage)
- [Deployment](#deployment)
- [Configuration](#configuration)
- [License](#license)


## Description

This project uses the Airbnb Open Data from Kaggle to build a machine learning model that predicts listing prices. As my first cloud deployment project, the goal is to create a lightweight prototype that resembles a production-style workflow.

The focus is not only on training the model, but also on understanding how to package and deploy it to the cloud. The prediction pipeline is containerized using Docker, uploaded to AWS ECR, and deployed as a serverless API using AWS Lambda and API Gateway. Users can send JSON requests and receive price predictions in real time.

This project covers the essential steps of an end-to-end ML workflow:

- Data preprocessing and feature engineering
- Model training and evaluation
- Packaging the prediction pipeline
- Containerization with Docker (multi-architecture build)
- Deploying the model to AWS Lambda
- Exposing the model via a REST API on API Gateway
- Designing clear JSON request/response structures for inference

The result is a practical, cloud-based prototype that demonstrates how machine learning models can be deployed and served using AWS infrastructure.

## Tech Stack

### Languages & Libraries
- Python
- NumPy
- Pandas
- Scikit-learn
- XGBoost
- Matplotlib
- Seaborn

### Cloud Services
- AWS Lambda
- AWS API Gateway
- AWS ECR
- AWS S3 
- AWS IAM

### Development Tools
- VSCode
- Git / GitHub
- AWS CLI
- Docker 

## Project Structure (Simplified / Key Files)

```plaintext
MLPROJECT/
├── artifacts/
│   ├── preprocessor.pkl
│   └── model.pkl
├── configs/
├── data/
├── src/
│   ├── components/
│   ├── prediction/
│   └── pipeline.py
├── Dockerfile
├── lambda_handler.py
├── requirements.txt
└── README.md
```

## Installation / Setup

```bash
# 1. Clone the repository
git clone https://github.com/wyhfelix05/mlproject
cd mlproject

# 2. Create a virtual environment
python -m venv mlenv

# Activate the environment
# On Windows
mlenv\Scripts\activate
# On macOS/Linux
source mlenv/bin/activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Optional: install development dependencies
pip install -r requirements-dev.txt
```

## Usage

### Start the API locally

After installing dependencies and activating your virtual environment, run:

```bash
uvicorn src.prediction.app:app --reload
```

## Deployment

```bash
# 1. Docker Deployment

# Build the Docker image locally or for multi-architecture
docker build --platform linux/amd64 -t mlproject-lambda:amd64 .

# Push the image to AWS ECR
aws ecr get-login-password --region <your-region> | docker login --username AWS --password-stdin <account-id>.dkr.ecr.<region>.amazonaws.com
docker tag mlproject-lambda:v1 <account-id>.dkr.ecr.<region>.amazonaws.com/mlproject-lambda:v1
docker push <account-id>.dkr.ecr.<region>.amazonaws.com/mlproject-lambda:v1
```

### AWS Lambda Deployment

Create or update Lambda function using the Docker image from ECR, configure memory, timeout, environment variables, and assign IAM role with S3 access.

Optional: Use Lambda Layers to separate large dependencies.

Configure Function URL or API Gateway to enable HTTP(S) access.

Ensure Lambda can access the S3 bucket and load the model dynamically.

## Configuration

The project uses YAML configuration files to manage paths, model parameters, and preprocessing settings.

### Key configuration files

- `configs/config.yaml`  
  Defines paths to input CSV files and output artifacts (e.g., preprocessed data, model output).

- `configs/params.yaml` *(if applicable)*  
  Stores hyperparameters for the ML models (e.g., CatBoost, XGBoost, training options).

## License

This project is licensed under the MIT License. You are free to use, modify, and distribute this project for personal or commercial purposes, provided that the original copyright notice and license are included.
