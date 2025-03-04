# Adaptive Concept Drift Detection and Retraining for Marine Vessel Exhaust Monitoring

This repository demonstrates, at a high level, a real-world end-to-end solution for handling concept drift and deep learning model retraining using marine vessel exhaust monitoring as the exemplar application. The outlined approach goes beyond generic drift detection by:
- **Enriching Sensor Data:** Incorporating additional signals (gas temperature, engine RPM, back-pressure, and derived gas velocity).
- **Integrating Domain-Specific Mechanistic Insights:** Deriving interpretable features (e.g., effective activation energy and effective gas constant) based on the Arrhenius equation and the ideal gas law.
- **Leveraging Reconstruction Losses:** Analyzing the LSTM’s reconstruction losses via advanced metrics like the Wasserstein distance to detect drift.
- **Robust MLOps Integration:** Orchestrating the pipeline with dynamic normalization, automated retraining, versioning, logging, and real-time dashboarding (using platforms like Databricks and MLflow).

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
- [MLOps Integration](#mlops-integration)
- [Frequently Asked Questions (FAQ)](#frequently-asked-questions-faq)
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
   - Derive mechanistic features (e.g., effective activation energy and effective gas constant) from the enriched data.
   - Combine these signals into a composite drift metric.

3. **Retraining Pipeline:**  
   - When the composite drift metric exceeds adaptive thresholds, automatically select a sliding window of recent, “clean” data.
   - Retrain the LSTM (modified to accept enriched features and dynamic normalization) using a custom loss function that penalizes high reconstruction errors.
   - Version, validate, and deploy the new model using MLflow.

4. **MLOps & Dashboarding:**  
   - Schedule automated jobs (via Databricks Jobs or Airflow) for drift monitoring and retraining.
   - Log dynamic normalization parameters, reconstruction error metrics, and mechanistic features.
   - Build dashboards for real-time visualization and transparency.

---

## Key Technical Concepts

### Dynamic Normalization

- **Concept:**  
  Update normalization parameters (mean, variance) on a rolling or batch-by-batch basis rather than using static, precomputed values.
- **Benefit:**  
  Adapts the model to shifts in the input distribution (e.g., due to soot buildup) and serves as an additional signal for drift.

### Composite Drift Detection

- **Concept:**  
  Fuse multiple signals—from reconstruction error metrics (mean, variance, Wasserstein distance) and mechanistic features (effective activation energy, effective gas constant)—into a single, robust drift metric.
- **Benefit:**  
  Provides a sensitive and reliable trigger for retraining by capturing average errors as well as changes in distribution shape and physical operating conditions.

### Mechanistic Feature Engineering

- **Concept:**  
  Leverage domain knowledge to derive features from sensor data:
  - **Arrhenius-Inspired Approach:**  
    Estimate an effective activation energy (\(E_a\)) that reflects changes in combustion or heat-transfer efficiency.
  - **Ideal Gas Law Approach:**  
    Derive an effective gas constant (\(R_{\text{eff}}\)) from gas temperature, engine RPM, and gas velocity.
- **Benefit:**  
  These interpretable, physically grounded signals complement statistical measures, offering deeper insight into the underlying process drift.

#### Mechanistic Feature Engineering Implementation

**A. Arrhenius-Inspired Approach**

1. **Data Collection:**  
   - Gather exhaust gas temperature (\(T\)) and a proxy for reaction rate (\(k\)) (e.g., normalized reconstruction error or another process-specific indicator).

2. **Linearization:**  
   - Transform the Arrhenius equation by taking the natural logarithm:
     \[
     \ln(k) = \ln(A) - \frac{E_a}{R} \cdot \frac{1}{T}
     \]
   - This yields a linear relationship between \(\ln(k)\) and \(1/T\).

3. **Estimation via Regression:**  
   - For each 15-minute batch, calculate \(1/T\) and \(\ln(k)\) and fit a linear regression model:
     - **Input:** \(x = 1/T\)
     - **Target:** \(y = \ln(k)\)
   - The slope \(m\) is used to estimate the effective activation energy:
     \[
     E_a = -m \times R
     \]
   - **Usage:**  
     A shift in \(E_a\) from its baseline indicates altered heat-transfer properties (e.g., increased insulation from soot).

**B. Ideal Gas Law Approach**

1. **Data Collection:**  
   - Collect gas temperature (\(T\)), engine RPM, and back-pressure (or gas velocity).

2. **Derivation:**  
   - Rearrange the Ideal Gas Law:
     \[
     R_{\text{eff}} = \frac{P \cdot V}{n \cdot T}
     \]
   - Here, \(P\) is estimated from back-pressure, \(V\) from gas velocity, and \(n\) is assumed constant.
   - **Estimation:**  
     Compute \(R_{\text{eff}}\) for each 15-minute batch and compare it to the baseline. Significant deviations signal a drift in the system's thermodynamic behavior.

**Integration:**  
- Log both \(E_a\) and \(R_{\text{eff}}\) over time and incorporate them into the composite drift metric alongside reconstruction error metrics.

### Modified Loss Function

- **Concept:**  
  Use a custom loss function (e.g., weighted MSE) that applies increased penalties for large reconstruction errors.
- **Benefit:**  
  Forces the model to learn a tighter representation of normal operations, making significant deviations a clearer signal of drift.

---

## Real-World Implementation Guidelines

### Feature Engineering and Data Consistency

- **Update the Feature Store:**  
  - Enrich both historical and live datasets with engine RPM, back-pressure, and derived gas velocity.
  - Compute and store separate normalization parameters for each feature.
  - **Ensure Consistency:** Both training and inference pipelines must use the same feature schema and normalization logic.

- **Data Alignment:**  
  - Correct for any signal lag (e.g., time-shift the RPM data if necessary) to maintain synchronized features.

### Retraining with New Features

- **Model Architecture Update:**  
  - Adjust the LSTM’s input layer to accept the enriched feature vector.
  - Optionally incorporate a dynamic normalization layer that updates in real time.
- **Training Pipeline:**  
  - Retrain the updated LSTM on the enriched dataset.
  - Validate that the new model generalizes to the current operating regime.

### Incorporating the Modified Loss Function

- **Design and Integrate:**  
  - Develop a custom loss function (such as a weighted MSE loss) that scales penalties based on the magnitude of reconstruction errors.
  - Replace the default loss in your training scripts and monitor training stability.

### Drift Detection and Retraining Automation

- **Monitoring:**  
  - Set up automated jobs to compute reconstruction error statistics (including the Wasserstein distance) and mechanistic features on 15-minute batches.
  - Combine these signals into a composite drift metric with adaptive thresholds.
- **Retraining Trigger:**  
  - Automatically trigger a retraining job when the composite drift metric consistently exceeds thresholds.
  - Select a sliding window of recent, “clean” data that reflects the new operating regime.
- **Versioning and Rollbacks:**  
  - Use MLflow for model versioning and logging drift metrics.
  - Build dashboards for continuous monitoring and support rapid rollback if needed.

### MLOps and Dashboarding

- **Logging and Visualization:**  
  - Log dynamic normalization parameters, reconstruction error metrics, and mechanistic coefficients.
  - Build interactive dashboards to visualize trends and provide transparency to stakeholders.
- **Orchestration:**  
  - Schedule periodic drift monitoring and retraining jobs using Databricks Jobs or Airflow.

---

## MLOps Integration

- **Automated Jobs:**  
  Use cloud orchestration (e.g., Databricks Jobs) to run drift detection every 15 minutes.
  
- **Version Control:**  
  Version both the enriched feature store and model artifacts with MLflow, ensuring traceability.
  
- **Dashboards:**  
  Create dashboards that visualize:
  - Rolling reconstruction error statistics.
  - Wasserstein distance trends.
  - Mechanistic parameters (effective activation energy and effective gas constant).
  - Dynamic normalization statistics.
  
- **Retraining and Rollback:**  
  Automatically trigger retraining based on the composite drift metric. Implement shadow deployments and maintain robust rollback mechanisms.

---

## Caveats and Considerations

- **Consistency is Key:**  
  Ensure that the new features and their normalization are applied consistently across historical and live data to avoid mismatches.
  
- **Dynamic Normalization Risks:**  
  While dynamic normalization adapts to changes, it may also mask drift if not logged separately.
  
- **Complexity Management:**  
  Incorporating mechanistic features and a modified loss function increases complexity. Each additional signal must be validated to ensure it reliably indicates drift.
  
- **Interpretability:**  
  Provide clear documentation and visualizations so that stakeholders understand how each signal (stati
