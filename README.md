# Smart Retail Supply Chain Analytics

### Inventory Optimization, Demand Forecasting & Pricing Intelligence

**AICTE | IBM SkillsBuild Data Analytics with AI Internship Project | BharatCares**

---

## Project Overview

This project presents an end-to-end retail supply chain analytics and machine learning workflow designed to analyze sales demand, inventory performance, product behavior, pricing, promotions, seasonality, and replenishment risk.

The project combines exploratory data analysis, business analytics, ABC-XYZ inventory segmentation, historical inventory risk assessment, machine learning-based demand forecasting, and forecast-driven inventory decision support.

The primary objective is not only to build a prediction model, but also to demonstrate how retail data can be transformed into useful supply chain insights for inventory planning and decision making.

> **Dataset Note:** The dataset used in this project is synthetic. Therefore, the findings demonstrate analytical methodology and decision-support techniques rather than the actual performance of a real retail organization.

---

## Project Objectives

The project focuses on the following objectives:

- Analyze historical retail sales and demand patterns
- Evaluate inventory levels and stock availability
- Identify product, category, store, and regional performance patterns
- Analyze pricing, discounts, promotions, and competitor pricing
- Study seasonal and external factors associated with demand
- Perform ABC product classification
- Perform XYZ demand variability classification
- Build an ABC-XYZ inventory segmentation framework
- Identify historical shortage, tight-stock, adequate, and excess inventory situations
- Build independent machine learning models for demand forecasting
- Compare multiple regression models using chronological validation
- Analyze model errors and forecasting limitations
- Compare the independently trained model with the dataset-provided demand forecast
- Evaluate how different forecast sources affect inventory decision support
- Generate reusable analytical outputs and trained model artifacts

---

## Dataset

**Dataset:** Retail Store Inventory Forecasting Dataset  
**Source:** Kaggle  
**Author:** Anirudh Singh Chauhan

Dataset link:

https://www.kaggle.com/datasets/anirudhchauhan/retail-store-inventory-forecasting-dataset

The original dataset contains approximately **73,100 records and 15 columns** covering retail inventory, demand, pricing, promotions, weather, seasonality, stores, products, and regions.

### Main Features

| Feature            | Description                      |
| ------------------ | -------------------------------- |
| Date               | Date of observation              |
| Store ID           | Store identifier                 |
| Product ID         | Product identifier               |
| Category           | Product category                 |
| Region             | Store region                     |
| Inventory Level    | Available inventory              |
| Units Sold         | Actual units sold                |
| Units Ordered      | Replenishment quantity           |
| Demand Forecast    | Dataset-provided demand forecast |
| Price              | Product selling price            |
| Discount           | Discount percentage/value        |
| Weather Condition  | Weather category                 |
| Holiday/Promotion  | Promotion or holiday indicator   |
| Competitor Pricing | Competitor price                 |
| Seasonality        | Seasonal category                |

`Units Sold` is used as the target variable for the independently trained machine learning models.

---

## Data Preparation

The workflow includes data quality checks, date processing, categorical and numerical feature preparation, analytical feature engineering, and chronological model splitting.

The final complete historical analytical period contains **73,000 records from 2022-01-01 through 2023-12-31**.

An additional **100 records dated 2024-01-01** contained incomplete information required by the modeling workflow and were excluded from chronological model evaluation.

To reduce potential information leakage:

- `Demand Forecast` was excluded from the independent ML feature set because its generation process is undocumented.
- `Units Ordered` was excluded because replenishment quantities may already contain information derived from previous demand expectations.

The dataset-provided `Demand Forecast` is instead evaluated separately as an **external benchmark**.

---

## Analytical Workflow

The project follows the workflow below:

**Data Loading → Data Quality Assessment → Exploratory Data Analysis → Business Analytics → ABC-XYZ Segmentation → Historical Inventory Risk Analysis → Feature Engineering → Chronological ML Training → Model Evaluation → Forecast Benchmark Comparison → Inventory Decision Validation**

---

## Exploratory & Business Analytics

The analytical stage investigates:

- Overall sales and demand patterns
- Product and category performance
- Store and regional performance
- Inventory availability
- Pricing behavior
- Competitor pricing
- Discounts and promotions
- Weather-related demand patterns
- Seasonal demand patterns
- Product-level demand variability

The emphasis of the project is primarily on business and supply chain analytics, with machine learning used as an additional forecasting and validation component.

---

## ABC-XYZ Inventory Segmentation

ABC-XYZ analysis combines product importance with demand variability.

### ABC Classification

Products are classified based on their contribution to cumulative sales value:

| Class | Number of Products |
| ----- | -----------------: |
| A     |                 15 |
| B     |                  3 |
| C     |                  2 |

### ABC-XYZ Segments

| Segment | Products |
| ------- | -------: |
| AX      |        6 |
| AY      |        5 |
| AZ      |        4 |
| BX      |        1 |
| BY      |        0 |
| BZ      |        2 |
| CX      |        0 |
| CY      |        1 |
| CZ      |        1 |

This segmentation helps distinguish strategically important and stable products from lower-value or highly variable products.

---

## Historical Inventory Risk Analysis

Historical inventory conditions are classified using the dataset-provided demand forecast and inventory position.

| Inventory Risk | Records |
| -------------- | ------: |
| Shortage       |       4 |
| Tight          |      37 |
| Adequate       |  22,287 |
| Excess         |  50,672 |

The historical analysis indicates that excess inventory represents the dominant inventory condition in this synthetic dataset.

A global historical safety buffer of approximately **7 units** was also derived from the 90th percentile of forecast error.

---

## Machine Learning Approach

Three regression models were evaluated:

1. Linear Regression
2. Random Forest Regressor
3. HistGradientBoosting Regressor

The target variable is:

`Units Sold`

A total of **21 model input features** were used before preprocessing:

- 15 numerical features
- 6 categorical features

After preprocessing and categorical encoding, the model matrix contained **57 processed features**.

---

## Chronological Train-Test Split

A chronological split was used instead of a random split to provide a more realistic forecasting evaluation.

| Dataset  | Period                   | Records |
| -------- | ------------------------ | ------: |
| Training | 2022-01-01 to 2023-09-30 |  63,800 |
| Testing  | 2023-10-01 to 2023-12-31 |   9,200 |

This prevents future observations from being randomly mixed into the training data.

---

## Model Performance

### Independently Trained Models

| Model                          |   Test MAE |  Test RMSE |    Test R² |
| ------------------------------ | ---------: | ---------: | ---------: |
| Linear Regression              | **68.946** | **87.923** | **0.3390** |
| Random Forest Regressor        |     70.645 |     89.020 |     0.3224 |
| HistGradientBoosting Regressor |     68.966 |     88.021 |     0.3376 |

Among the independently trained models, **Linear Regression was selected** because it achieved the lowest chronological test MAE and RMSE and showed a very small train-test performance gap.

The Random Forest model produced stronger training performance but weaker test performance, indicating greater overfitting.

---

## External Forecast Benchmark

The dataset also contains a pre-existing `Demand Forecast` field.

For the same chronological test period, the supplied forecast produced:

| Forecast Source                       |    MAE |   RMSE |     R² |
| ------------------------------------- | -----: | -----: | -----: |
| Selected ML Model — Linear Regression | 68.946 | 87.923 | 0.3390 |
| Dataset-Provided Demand Forecast      |  8.347 | 10.035 | 0.9914 |

The supplied forecast performs substantially better numerically.

However, its generation methodology is not documented. Therefore, it is treated as an **external benchmark rather than a model trained by this project**, and the comparison should not be interpreted as a fully like-for-like model competition.

---

## Model Error Analysis

Further error analysis revealed an important limitation of the selected model.

For test observations where actual demand exceeded **300 units**:

- Number of records: **911**
- Average actual units sold: **363.47**
- Average ML prediction: **213.18**
- MAE: **150.29**
- Underforecast rate: **100%**

This shows that the Linear Regression model tends to regress toward the average and substantially underpredict high-demand observations.

The trained model should therefore be considered a **baseline demand forecasting model**, not a production-ready forecasting system.

---

## Inventory Decision-Support Validation

Forecast quality was also evaluated from an inventory decision perspective.

### Test-Period Inventory Risk

| Risk Level | Supplied Forecast | ML Forecast |
| ---------- | ----------------: | ----------: |
| Shortage   |                 0 |           0 |
| Tight      |               308 |           0 |
| Adequate   |             2,502 |           0 |
| Excess     |             6,390 |       9,200 |

The independently trained ML model classified all test observations as excess inventory under the project decision rules.

This result demonstrates an important supply chain lesson: **forecasting performance must be evaluated not only using statistical metrics, but also by examining its downstream operational decisions.**

Because the baseline ML model underpredicts high demand and produces unrealistic inventory-risk behavior, the historical inventory analysis remains the primary source for inventory insights in this project.

---

## Feature Influence

Linear Regression coefficient analysis was used to inspect model feature influence.

The strongest coefficient by absolute magnitude was associated with the engineered:

`Competitor_Price_Ratio`

with a coefficient of approximately:

`-4.1695`

This represents a model association and should **not** be interpreted as evidence of a causal relationship.

---

## Why LSTM Was Not Used

LSTM was considered but deliberately excluded from the final modeling workflow.

The current project uses structured daily retail records without a dedicated lagged sequence architecture. Classical machine learning models provide a clearer and more appropriate baseline for the current internship scope.

A future version could build product-store level time-series sequences with lag features, rolling statistics, and dedicated forecasting models before evaluating LSTM or other deep-learning approaches.

---

## Project Architecture

The final project contains six major analytical components:

1. Exploratory Data Analysis
2. ABC-XYZ Inventory Segmentation
3. Historical Inventory Risk Analysis
4. Independent Machine Learning Forecasting
5. External Forecast Benchmarking
6. Forecast-Based Inventory Decision Validation

---

## Repository Structure

```text
smart-retail-supply-chain/
│
├── data/
│   └── retail_cleaned_data.csv
│
├── model/
│   ├── linear_regression_model.joblib
│   ├── preprocessor.joblib
│   ├── retail_demand_model_bundle.joblib
│   ├── processed_feature_names.csv
│   └── model_metadata.json
│
├── outputs/
│   ├── executive_kpis.csv
│   ├── abc_xyz_product_segmentation.csv
│   ├── historical_inventory_risk_summary.csv
│   ├── forecast_accuracy_comparison.csv
│   ├── ml_test_predictions_and_errors.csv
│   └── additional analytical outputs
│
├── src/
│   └── test_model.py
│
├── SayanGhorai_SmartRetailSupplyChain.ipynb
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Generated Analytical Outputs

The notebook exports reusable analytical results including:

- Executive KPI summary
- ABC classification summary
- ABC-XYZ product segmentation
- ABC-XYZ segment summary
- Historical inventory KPIs
- Historical inventory risk records
- Historical inventory risk summary
- Product inventory risk summary
- Priority shortage records
- ML test predictions and errors
- Forecast accuracy comparison
- Forecast inventory-risk comparison
- Forecast replenishment comparison
- Forecast decision-support comparison
- Linear Regression feature influence
- Final project findings
- Final project architecture
- Export manifest

These files allow the analysis to be reused for reporting, validation, and future dashboard development.

---

## Saved Model Artifacts

The selected model and preprocessing pipeline are exported to the `model/` directory.

The primary deployment artifact is:

```text
model/retail_demand_model_bundle.joblib
```

The saved model package includes the trained Linear Regression model and preprocessing components required to reproduce predictions.

A portability test script is included in:

```text
src/test_model.py
```

---

## Technologies Used

- Python 3.12
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Joblib
- Jinja2
- Jupyter Notebook
- Kaggle
- Visual Studio Code
- Git
- GitHub

---

## Installation

Clone the repository:

```bash
git clone https://github.com/SayanGhorai/smart-retail-supply-chain.git
cd smart-retail-supply-chain
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Project

Open:

```text
SayanGhorai_SmartRetailSupplyChain.ipynb
```

Run the notebook cells sequentially to reproduce the analytical workflow.

To validate the saved machine learning model package:

```bash
python src/test_model.py
```

---

## Requirements

The project dependencies are defined in `requirements.txt`:

```text
pandas==2.3.3
numpy==2.0.2
matplotlib==3.11.2
scikit-learn==1.6.1
joblib==1.5.3
Jinja2==3.1.6
```

---

## Key Conclusions

The project demonstrates that retail supply chain analysis benefits from combining descriptive analytics, inventory segmentation, forecasting, and operational validation.

The main findings include:

- ABC-XYZ segmentation provides a structured method for prioritizing inventory management.
- Historical inventory analysis identified excess inventory as the dominant condition in the synthetic dataset.
- Chronological validation provides a more realistic evaluation than random train-test splitting for time-dependent retail data.
- Linear Regression provided the best generalization among the independently trained models evaluated.
- The selected baseline ML model has significant limitations for high-demand observations.
- The dataset-provided forecast performs much better statistically but cannot be treated as an independently trained project model because its generation methodology is undocumented.
- Forecast models should be evaluated using both statistical accuracy and their downstream inventory decisions.
- High forecast accuracy alone is not sufficient unless the resulting operational recommendations are also reasonable.

---

## Limitations

This project has several important limitations:

- The dataset is synthetic and does not represent a real retailer.
- The generation process of the supplied demand forecast is undocumented.
- The independent ML models do not use dedicated lagged time-series sequences.
- Linear Regression underpredicts high-demand observations.
- Model coefficients represent statistical associations and not causal effects.
- Inventory decision rules are analytical approximations and are not optimized against real-world service levels, supplier lead times, or business costs.

---

## Future Improvements

Potential future extensions include:

- Lag-based demand features
- Rolling demand statistics
- Product-store level forecasting models
- Time-series cross-validation
- XGBoost or LightGBM comparison
- Dedicated time-series models
- LSTM or sequence-based deep learning after appropriate sequence preparation
- Dynamic safety-stock calculations
- Supplier lead-time integration
- Service-level optimization
- Interactive Streamlit dashboard
- Automated inventory alerts and replenishment recommendations

---

## Internship Submission

This project was developed as part of the:

**AICTE | IBM SkillsBuild Data Analytics with AI Internship Program — BharatCares**

The submission package includes:

- Jupyter Notebook
- Requirements File
- Project Report
- README Documentation

---

## Author

**Sayan Ghorai**

AICTE | IBM SkillsBuild Data Analytics with AI Internship Project

---

## Disclaimer

This project is intended for educational and analytical demonstration purposes.

Because the source dataset is synthetic, the results should not be interpreted as actual commercial retail performance or used directly for real-world inventory decisions without further validation.
