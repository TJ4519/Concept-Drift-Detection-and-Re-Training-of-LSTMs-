# Adaptive Concept Drift Detection and Retraining for Marine Vessel Exhaust Monitoring

This repository demonstrates an end-to-end solution for handling concept drift in marine vessel exhaust monitoring systems. Our approach goes beyond generic drift detection by enriching sensor data (e.g., gas temperature, engine RPM, back-pressure, derived gas velocity), integrating domain-specific mechanistic insights, and using the LSTM’s reconstruction losses—analyzed via advanced metrics like the Wasserstein distance—to detect drift. 

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [Proposed Solution Architecture](#proposed-solution-architecture)
- [Key Technical Concepts](#key-technical-concepts)
  - [Dynamic Normalization](#dynamic-normalization)
  - [Composite Drift Detection](#composite-drift-detection)
  - [Mechanistic Feature Engineering](#mechanistic-feature-engineering)
  - [Modified Loss Function](#modified-loss-function)
- [Expert Commentary and Rationale](#expert-commentary-and-rationale)
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

Marine vessel exhaust monitoring is critical for early fault detection. Deep-learning models—such as LSTMs—are trained on sensor data (e.g., gas temperature) to predict normal operating conditions. Reconstruction errors (the difference between predicted and observed gas temperatures) are used to flag anomalies that may precede catastrophic failures (e.g., fires due to excessive soot buildup).

However, gradual changes in the exhaust system—such as the insulating effect of accumulating soot—cause the underlying data distribution to drift over time. This drift not only reduces model accuracy but can also mask critical anomalies. Our solution addresses these challenges by:

- **Enriching the Feature Set:** Incorporating additional signals (engine RPM, back-pressure, derived gas velocity) and deriving mechanistic features.
- **Robust Drift Detection:** Monitoring the full reconstruction error distribution (using metrics like the Wasserstein distance) alongside mechanistic parameters.
- **Automated Retraining:** Triggering retraining using a composite drift signal that selects a new dataset representing the current “normal.”
- **MLOps Integration:** Orchestrating the entire pipeline with robust versioning, logging, dashboarding, and automated retraining.

---

## Proposed Solution Architecture

### Overview

1. **Feature Store & Data Enrichment:**  
   - Enrich sensor data by integrating engine RPM, back-pressure, and derived gas velocity.
   - Use dynamic normalization to compute per-feature statistics (mean, variance) on a rolling basis.

2. **Drift Monitoring Module:**  
   - Compute reconstruction error statistics (mean, variance) over 15-minute batches.
   - Use the Wasserstein distance to compare the current reconstruction error distribution with a historical baseline.
   - Derive mechanistic features (e.g., effective activation energy or effective gas constant) from the enriched data.
   - Combine these signals into a composite drift metric.

3. **Retraining Pipeline:**  
   - When the composite drift metric exceeds adaptive thresholds, automatically select a sliding window of recent, “clean” data.
   - Retrain the LSTM (modified to accept enriched features and dynamic normalization) using a custom loss function that penalizes high reconstruction errors.
   - Version, validate, and deploy the new model using MLflow.

4. **MLOps & Dashboarding:**  
   - Schedule automated jobs (using Databricks Jobs or Airflow) for drift monitoring and retraining.
   - Log dynamic normalization parameters, reconstruction error metrics, and mechanistic features.
   - Build dashboards for real-time visualization and transparency.

---

## Key Technical Concepts

### Dynamic Normalization

- **Concept:**  
  Update normalization parameters (mean, variance) on a rolling or batch-by-batch basis, rather than using static, precomputed values.
- **Benefit:**  
  Allows the model to adapt to shifts in the input distribution (e.g., due to soot buildup) and provides an extra drift signal.

### Composite Drift Detection

- **Concept:**  
  Fuse multiple signals—from reconstruction error metrics (mean, variance, Wasserstein distance) and mechanistic features (effective activation energy, effective gas constant)—into a single, robust drift metric.
- **Benefit:**  
  Offers a sensitive and reliable trigger for retraining by capturing not only average errors but also distribution shape and physical insights.

### Mechanistic Feature Engineering

- **Concept:**  
  Leverage domain knowledge to derive features from sensor data:
  - **Arrhenius-Inspired Approach:** Estimate an effective activation energy that reflects changes in combustion or heat-transfer efficiency.
  - **Ideal Gas Law Approach:** Derive an effective gas constant from gas temperature, RPM, and gas velocity.
- **Benefit:**  
  Provides interpretable, physically grounded signals of drift that complement purely statistical measures.

### Modified Loss Function

- **Concept:**  
  Use a custom loss (e.g., weighted MSE) that penalizes high reconstruction errors more heavily, forcing the model to maintain a tight error distribution under normal conditions.
- **Benefit:**  
  Enhances anomaly detection, making deviations (and hence concept drift) more pronounced.

---

## Expert Commentary and Rationale

### Why Not Use Generic Drift Detection Tests?

- **Generic Methods’ Limitations:**  
  Generic drift tests (e.g., KS test, Page-Hinkley) are designed for broad applications. They typically compare input data distributions rather than model-specific signals like reconstruction losses. They do not capture the rich, domain-specific information inherent in the reconstruction error distribution produced by an LSTM.
  
- **Our Tailored Approach:**  
  By focusing on reconstruction losses, our approach directly reflects the LSTM’s performance. Additionally, incorporating mechanistic features grounds the analysis in the physics of the exhaust system. This combination leads to more sensitive and interpretable drift detection—crucial when drift indicates safety-critical issues (e.g., soot buildup leading to fire risk).

### Why Use the Wasserstein Distance?

- **Holistic Comparison:**  
  The Wasserstein distance measures the “cost” to transform the baseline reconstruction error distribution into the current one. It captures changes in mean, spread, and shape (skewness, kurtosis) that simpler metrics might miss.
  
- **Sensitivity to Drift:**  
  Even subtle changes in the error distribution—such as increased outliers or fat tails—will result in a higher Wasserstein distance, serving as an early warning signal.
  
- **Robustness:**  
  When used in a composite drift metric alongside mechanistic features, it offers robust evidence for when retraining is necessary.

---

## Real-World Implementation Guidelines

### Feature Engineering and Data Consistency

- **Update the Feature Store:**  
  - Enrich both historical and live datasets with new sensor signals.
  - Compute per-feature normalization parameters and ensure consistent preprocessing across training and inference.
  
- **Data Alignment:**  
  - Adjust for any signal lag (e.g., time-shift RPM) to ensure features are synchronized.

### Retraining with New Features

- **Model Architecture Update:**  
  - Modify the LSTM’s input layer to handle the enriched feature vector.
  - Integrate a dynamic normalization layer (if desired) to update normalization in real time.
  
- **Training Pipeline:**  
  - Train the updated LSTM on the enriched dataset.
  - Validate performance to ensure that the model generalizes under the new operating conditions.

### Incorporating the Modified Loss Function

- **Design the Loss Function:**  
  - Develop a custom loss function (e.g., a weighted MSE loss) that scales up penalties for large reconstruction errors.
  
- **Integration:**  
  - Replace the default loss in your training scripts.
  - Monitor training to ensure stability and adjust hyperparameters as needed.

### Drift Detection and Retraining Automation

- **Monitoring:**  
  - Create automated jobs that compute reconstruction error metrics (including the Wasserstein distance) and mechanistic features on 15-minute batches.
  - Combine these signals into a composite drift metric with adaptive thresholds.
  
- **Retraining Trigger:**  
  - When the composite drift metric exceeds threshold values over several consecutive batches, automatically select a sliding window of recent “clean” data for retraining.
  
- **Versioning and Rollbacks:**  
  - Use MLflow for model versioning and logging drift metrics.
  - Set up dashboards for continuous monitoring and easy rollback.

### MLOps and Dashboarding

- **Logging and Visualization:**  
  - Log dynamic normalization parameters, reconstruction errors, and derived mechanistic coefficients.
  - Build interactive dashboards to show trends and alert stakeholders when drift is detected.
  
- **Orchestration:**  
  - Schedule periodic drift monitoring and retraining jobs using Databricks Jobs or Airflow.

---

## Simulation Example: Adaptive Drift Detection

For a demonstration, we provide a simple Python simulation (`simulate_drift.py`) that shows how a baseline reconstruction error distribution gradually drifts due to factors such as increased insulation from soot buildup. The simulation computes and visualizes key metrics (mean, variance, skew, kurtosis, and Wasserstein distance) to mimic a real-time drift detection dashboard.

*(See the Simulation Example section in this repository for the code and detailed commentary.)*

---

## Future Work

- Refine mechanistic feature derivation using real sensor data.
- Experiment with alternative adaptive normalization techniques and modified loss functions.
- Expand the drift detection module with additional statistical tests and ensemble methods.
- Enhance dashboard interactivity with real-time alerts and detailed retraining logs.

---

## Conclusion

This repository presents a robust, adaptive solution for concept drift detection and retraining in marine vessel exhaust monitoring. By enriching the feature set, leveraging dynamic normalization, and using advanced metrics such as the Wasserstein distance on reconstruction losses—augmented with domain-specific mechanistic features—we provide a comprehensive, end-to-end pipeline. Our approach is tightly integrated with MLOps best practices, ensuring that the system is both scalable and interpretable, and ready to handle real-world challenges in safety-critical environments.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Contact

For questions or suggestions, please contact [Your Name](mailto:your.email@example.com).
