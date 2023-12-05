# Credit Card Fraud Detection Using Custom Models

## Introduction

This project, Anti-Scamming Predictor, aims to detect credit card fraud using custom machine learning models. It provides a Streamlit-based GUI for easy interaction with the predictive models.

## Prerequisites

Before setting up the project, ensure you have Anaconda installed on your system. You can download it from Anaconda's website. (https://www.anaconda.com/download)

## Data File Preparation

The project uses a dataset named creditcard.csv which is essential for the fraud detection models. This dataset can be obtained in two ways:

- Using Provided Zip File:
    - Locate the creditcard.csv.zip file in the project directory.
    - Unzip this file to extract the creditcard.csv file.
    - Ensure that the extracted CSV file is in the same directory as your project files for easy access by the application.
- Downloading from Kaggle:
    - Alternatively, you can download the dataset directly from Kaggle at this link (https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).
    - After downloading, unzip the file and place creditcard.csv in the project's working directory.

## Setting Up the Conda Virtual Environment

Follow these steps to set up the conda environment:

- conda create --name myenv
- conda activate myenv
- conda install pip
- pip install -r requirements.txt
- streamlit run Group4_GUI.py

## Verification of Setup

After installation, you can verify the setup by running 'conda list' in your environment to check if all required packages are installed.

## Usage Instructions

Once the environment is set up and the application is running, navigate to the local URL provided by Streamlit in your browser to interact with the application.

## Example Input transaction 
- 0.0,134.0,0.0,123.0,0.0,0.0,0.0,0.0,0.0,-50.0,132.0,-5.0,0.0,-6.0,0.0,0.0,-7.0,0.0,335.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,3333.03
- -1.359807134,-0.072781173,2.536346738,1.378155224,-0.33832077,0.462387778,0.239598554,0.098697901,0.36378697,0.090794172,-0.551599533,-0.617800856,-0.991389847,-0.311169354,1.468176972,-0.470400525,0.207971242,0.02579058,0.40399296,0.251412098,-0.018306778,0.277837576,-0.11047391,0.066928075,0.128539358,-0.189114844,0.133558377,-0.021053053,149.62
