# Bridge Risk Intelligence Prototype

Machine learning prototype for **bridge deterioration risk prioritization**, developed from a Career Experience Practicum project with **IHI**.

The goal is to help analysts and engineers screen bridge inspection records faster and create a **priority list for review**.

> This project supports early screening only. It does not replace certified bridge inspection or engineering judgement.

---

## Project Overview

```mermaid
flowchart TD
    A[Problem<br/>Many bridge records are reviewed manually] --> B[Solution<br/>ML based risk ranking prototype]
    B --> C[Expected Result<br/>Priority list for engineer review]
```

---

## Practicum Context

```mermaid
flowchart TD
    A[Career Experience Practicum] --> B[IHI and Aster project]
    B --> C[Role: Data Analyst]
    C --> D[Work: Bridge risk analysis and proposal design]
    D --> E[Result: Team ranked 3rd<br/>Score: 95 / 100]
```

---

## Input and Output

```mermaid
flowchart TD
    A[Input<br/>California bridge inspection files<br/>CA21 to CA25] --> B[Process<br/>Clean data<br/>Match yearly records<br/>Create target<br/>Train model<br/>Rank risk]
    B --> C[Output<br/>Bridge priority list<br/>Model metrics<br/>Dashboard prototype]
```

---

## Machine Learning Task

```mermaid
flowchart TD
    A[Current year bridge data] --> B[Features<br/>Age<br/>Traffic<br/>Geometry<br/>Condition<br/>Ownership]
    B --> C[Model]
    C --> D[Prediction<br/>Will LOWEST_RATING decrease next year?]
    D --> E[Risk score]
    E --> F[Priority ranking]
```

---

## Validation Design

```mermaid
flowchart LR
    A[Train<br/>2021 to 2022<br/>2022 to 2023] --> B[Validation<br/>2023 to 2024]
    B --> C[Test<br/>2024 to 2025]
```

A temporal split is used because the model should be tested on future bridge records, not randomly mixed data.

---

## Models

| Model | Purpose |
|---|---|
| Logistic Regression | Simple baseline and explainable ranking |
| Decision Tree | Lightweight nonlinear comparison |

The best model is selected using **validation PR AUC**, because bridge deterioration is a rare event.

---

## Current Result

Selected model:

```text
Logistic Regression
```

| Metric | Value |
|---|---:|
| ROC AUC | 0.7197 |
| PR AUC | 0.0879 |
| Precision | 0.0900 |
| Recall | 0.3586 |
| F1 | 0.1438 |

Interpretation: the model has useful ranking signal, but it is not strong enough for engineering decisions alone.

---

## Ranking Value

```mermaid
flowchart TD
    A[All bridges] --> B[Model scores risk]
    B --> C[Top 1 percent<br/>about 3.80x lift]
    B --> D[Top 5 percent<br/>about 2.93x lift]
    B --> E[Top 10 percent<br/>about 2.63x lift]
```

This means the model can help reviewers start from a smaller, higher-risk group.

---

## Project Structure

```text
.
├── app.py
├── requirements.txt
├── notebooks/
│   └── 01_bridge_risk_prototype.ipynb
├── src/
│   └── bridge_risk_pipeline.py
├── data/
│   ├── raw/
│   └── processed/
├── outputs/
├── reports/
├── models/
└── docs/
```

---

## How to Run

```bash
python -m venv .venv
source .venv/Scripts/activate
pip install -r requirements.txt
python src/bridge_risk_pipeline.py --fast
streamlit run app.py
```

---

## Main Files

| File | Purpose |
|---|---|
| `notebooks/01_bridge_risk_prototype.ipynb` | Presentation notebook |
| `src/bridge_risk_pipeline.py` | Main data and ML pipeline |
| `app.py` | Streamlit dashboard prototype |
| `outputs/` | Priority list outputs |
| `reports/` | Metrics and analysis reports |
| `models/` | Saved trained model |

---

## Limitation

This prototype predicts possible changes in reported bridge condition ratings.  
It does **not** predict bridge collapse, certify safety, or replace professional inspection.

---

## Future Work

```mermaid
flowchart TD
    A[More historical years] --> B[Better probability calibration]
    C[Maintenance history] --> B
    D[Climate and traffic trends] --> B
    B --> E[Engineer validation]
    E --> F[Secure internal dashboard]
```
