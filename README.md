# Adaptive Concept Drift Detection and Retraining for Marine Vessel Exhaust Monitoring

This repository demonstrates an end-to-end solution for handling concept drift in marine vessel exhaust monitoring systems. Our approach enriches sensor data (e.g., gas temperature, engine RPM, back-pressure, derived gas velocity) and uses an LSTM model to predict normal operations. Reconstruction losses are monitored and analyzed using the Wasserstein distance to detect drift, which then triggers retraining. The solution is implemented with robust MLOps practices on a cloud platform (e.g., Databricks) and versioned using MLflow.

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [Proposed Solution Architecture](#proposed-solution-architecture)
- [Key Technical Concepts](#key-technical-concepts)
  - [Dynamic Normalization](#dynamic-normalization)
  - [Composite Drift Detection](#composite-drift-detection)
  - [Mechanistic Feature Engineering](#mechanistic-feature-engineering)
  - [Modified Loss Function](#modified-loss-function)
- [Real-World Implementation Guidelines](#real-world-implementation-guidelines)
- [Simulation Example: Adaptive Drift Detection](#simulation-example-adaptive-drift-detection)
- [MLOps Integration](#mlops-integration)
- [Caveats and Considerations](#caveats-and-considerations)
- [Getting Started](#getting-started)
- [Future Work](#future-work)
- [Conclusion](#conclusion)
- [License](#license)
- [Contact](#contact)

---

## Problem Statement

Marine vessel exhaust monitoring is crucial for early fault detection. Our current LSTM model predicts gas temperature based on sensor data and flags anomalies via reconstruction losses. However, gradual changes—such as soot buildup that alters exhaust heat-transfer properties—cause the data distribution to drift, leading to outdated models and unreliable anomaly detection.

**Our Goals:**
- **Enrich the Feature Set:** Incorporate additional signals (engine RPM, back-pressure, derived gas velocity) and derive mechanistic features to capture the physical state of the exhaust.
- **Robust Drift Detection:** Monitor the entire reconstruction loss distribution by computing the Wasserstein distance between the current and baseline distributions.
- **Automated Retraining:** When a composite drift signal (combining statistical and mechanistic features) indicates significant drift, automatically select a new training dataset that reflects the new operating regime and retrain the LSTM.
- **MLOps Integration:** Orchestrate this entire pipeline using robust versioning, logging, and dashboarding practices.

---

## Proposed Solution Architecture

### Overview

1. **Feature Store & Data Enrichment:**  
   - Enrich sensor data by adding engine RPM, back-pressure, and derived gas velocity.
   - Apply dynamic normalization per feature (each with its own mean/variance).

2. **Drift Monitoring Module:**  
   - Compute reconstruction error statistics (mean, variance) on 15-minute batches.
   - Use the Wasserstein distance to measure how the current error distribution differs from a historical baseline.
   - Derive mechanistic features (e.g., effective activation energy, effective gas constant) from enriched data.
   - Combine these signals into a composite drift metric.

3. **Retraining Pipeline:**  
   - When drift is detected (composite metric exceeds thresholds), select a sliding window of recent data (with anomaly filtering) that reflects the new “normal.”
   - Retrain the LSTM (modified to accept enriched features and dynamic normalization) using a custom loss function that penalizes high reconstruction errors.
   - Version, validate, and deploy the new model using MLflow.

4. **MLOps & Dashboarding:**  
   - Use automated jobs (via Databricks Jobs or Airflow) for drift monitoring and retraining.
   - Log dynamic normalization parameters, reconstruction error metrics, and mechanistic features.
   - Build dashboards to visualize trends and provide transparency for stakeholders.

---

## Key Technical Concepts

### Dynamic Normalization

- **Concept:** Compute per-feature normalization parameters (mean and variance) on a sliding window basis rather than using static, precomputed values.
- **Benefit:** Allows the model to adapt to changes in the data distribution and provides additional drift signals.

### Composite Drift Detection

- **Concept:** Aggregate multiple signals—statistical metrics from reconstruction errors (mean, variance, Wasserstein distance) and mechanistic features—into a composite metric.
- **Benefit:** A more robust indicator that a shift in operating conditions (e.g., due to soot buildup) has occurred.

### Mechanistic Feature Engineering

- **Concept:** Use domain knowledge to derive features from sensor data. For example:
  - **Arrhenius-Inspired Approach:** Estimate an effective activation energy.
  - **Ideal Gas Law Approach:** Derive an effective gas constant from gas temperature, RPM, and gas velocity.
- **Benefit:** Provides physically interpretable signals of drift.

### Modified Loss Function

- **Concept:** Use a custom loss function (e.g., weighted MSE) that penalizes high reconstruction errors more heavily, ensuring a tight error distribution under normal conditions.
- **Benefit:** Enhances anomaly detection and makes drift signals more pronounced.

---

## Real-World Implementation Guidelines

### Feature Engineering and Data Consistency

- **Update the Feature Store:**  
  - Enrich historical and live datasets with additional sensor signals (engine RPM, back-pressure, derived gas velocity).
  - Compute and store separate normalization statistics for each feature.
  - **Important:** Ensure that both training and inference pipelines use the same feature schema and normalization parameters.

- **Data Alignment:**  
  - Address any lag issues (e.g., time-shift the RPM signal if necessary) to maintain consistency across features.

### Retraining with New Features

- **Model Architecture Update:**  
  - Modify the LSTM’s input layer to accept the enriched feature vector.
  - Optionally integrate a dynamic normalization layer to adapt to changing distributions.
- **Training Pipeline:**  
  - Use the updated feature store to train the LSTM.
  - Validate that the new model generalizes on the enriched data.

### Incorporating the Modified Loss Function

- **Design the Loss Function:**  
  - Develop a custom loss function that multiplies the standard MSE by a weight function \(w(e)\) that increases for larger errors.
- **Integration:**  
  - Replace the default loss in your training script with the custom loss.
  - Monitor training stability and adjust hyperparameters to prevent gradient instability.

### Drift Detection and Retraining Automation

- **Monitoring:**  
  - Create a Databricks job to compute rolling statistics on reconstruction losses and calculate the Wasserstein distance between the current and baseline error distributions.
  - Combine these signals with mechanistic feature trends into a composite drift metric.
- **Retraining Trigger:**  
  - Automatically trigger a retraining job when the composite metric exceeds adaptive thresholds.
  - Select a recent sliding window of “clean” data that reflects the new operating regime.
- **Versioning and Rollbacks:**  
  - Use MLflow for model versioning and maintain logs of drift metrics and normalization parameters.
  - Set up dashboards for continuous monitoring.

---


