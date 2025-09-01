# Anomaly Detection Framework for Financial and IoT Data

This project implements and demonstrates a robust framework for anomaly detection using three distinct machine learning techniques. It applies these methods to two synthetic, real-world datasets: one simulating financial transactions to detect fraud and another simulating IoT sensor data to identify equipment faults.

## 🚀 Features

- **Multiple Unsupervised Models:** Implements and compares three powerful anomaly detection algorithms:
    - **Isolation Forest:** A tree-based model that excels at isolating outliers.
    - **Local Outlier Factor (LOF):** A density-based algorithm that identifies anomalies by measuring local deviation.
    - **Autoencoder:** A neural network-based approach that detects anomalies based on high reconstruction error.
- **Real-World Use Cases:**
    - **Financial Fraud Detection:** Identifies irregular transactions in a synthetic financial dataset.
    - **IoT Sensor Monitoring:** Detects faulty readings from simulated IoT equipment sensors.
- **Modular & Reusable Code:** The project is structured with a `src/` directory containing clean, documented, and reusable functions for data preprocessing, model training, and evaluation.
- **End-to-End Demonstration:** A comprehensive Jupyter notebook provides a clear walkthrough of the entire process, from data loading to identifying and analyzing anomalies.

<!-- ## 📂 Project Structure

A clean and scalable project structure designed for clarity and maintainability.

anomaly-detection-ml/
├── .gitignore                # Standard file to ignore Python artifacts
├── README.md                 # You are here!
├── requirements.txt          # Project dependencies
├── data/
│   ├── finance_transactions.csv  # Synthetic financial transaction data
│   └── iot_sensor_readings.csv   # Synthetic IoT sensor data
├── notebooks/
│   └── anomaly_detection_demo.ipynb  # Main demonstration notebook
├── results/
│   └── README.md               # Explanation of the results directory
└── src/
├── init.py
├── data_utils.py           # Data loading and preprocessing functions
└── anomaly_models.py       # Implementations of the detection models -->


## ⚡ Setup & Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd anomaly-detection-ml
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required dependencies from `requirements.txt`:**
    ```bash
    pip install -r requirements.txt
    ```

## ▶️ How to Run the Demo

The primary workflow and demonstrations are available in the Jupyter Notebook.

1.  **Start the Jupyter Notebook server:**
    ```bash
    jupyter notebook
    ```
2.  **Open the demo notebook:**
    Navigate to `notebooks/anomaly_detection_demo.ipynb` to see the end-to-end workflow applying all three models to both datasets.

## 💡 Project Motivation
This project was developed to create a practical, hands-on framework for one of the most critical unsupervised learning tasks in the industry: anomaly detection. Inspired by real-world challenges in finance and industrial operations, this repository serves as a personal exploration into building reliable and interpretable systems for identifying rare and suspicious events. The goal is to provide a clean, reusable, and well-documented starting point for tackling similar problems.

---
📝 **Author:** Lakshay Naresh Ramchandani