# Bridge Risk Intelligence Prototype

## 1. Project summary

This project is a machine learning # Bridge Risk Intelligence Prototype

Prototype phân tích và dự đoán rủi ro xuống cấp cầu (bridge deterioration risk), được xây dựng trong khuôn khổ Career Experience Practicum với **IHI**. Notebook chính: `01_bridge_risk_prototype.ipynb`.

Mục tiêu: tự động hóa bước sàng lọc dữ liệu kiểm định cầu và tạo ra danh sách ưu tiên (priority list) để kỹ sư review, thay vì phải rà soát thủ công toàn bộ dữ liệu.

---

## 🎓 Career Practicum

| | |
|---|---|
| **Program** | Career Experience Practicum |
| **Enterprise Experience** | IHI & Aster internship-based project |
| **Role** | Data Analyst |
| **Work** | Bridge risk analysis and business proposal development |
| **Result** | Team ranked **3rd**, received **95/100** |

**Reflection:** This experience helped me understand enterprise business problems, infrastructure risk, and data-driven proposal design.

---

## 📷 Project Photo

<!--
Đặt ảnh vào thư mục assets/ (ví dụ: assets/team-photo.jpg) rồi thay đường dẫn bên dưới.
-->

![Team photo — attach your photo here](./assets/team-photo.jpg)

*(Thay ảnh placeholder ở trên bằng ảnh thật của team/dự án, ví dụ ảnh present hoặc ảnh nhóm.)*

---

## Project Structure (expected)

```
.
├── 01_bridge_risk_prototype.ipynb   # Notebook trình bày (presentation)
├── src/
│   └── bridge_risk_pipeline.py      # Code chính, comment bằng tiếng Việt
├── data/                            # Dữ liệu đầu vào (raw)
├── outputs/                         # Priority list (kết quả)
├── reports/                         # Thống kê target, đánh giá model
├── models/                          # Model đã train
└── assets/                          # Ảnh cho README (team photo, v.v.)
```

## How to Run

1. Cài các thư viện cần thiết: `pandas`, `plotly`, và các thư viện dùng trong `src/bridge_risk_pipeline.py`.
2. Mở `01_bridge_risk_prototype.ipynb` bằng Jupyter.
3. Chạy tuần tự từ trên xuống — mỗi phần đều có **note ôn tập** (🎯 Mục tiêu / 📥 Input / 📤 Output) ngay phía trên để dễ nhớ vì sao cần bước đó.

## Note

Đây là công cụ hỗ trợ **ưu tiên kiểm tra**, không phải công cụ chứng nhận an toàn cầu.prototype for **bridge deterioration risk prioritization**. It uses California bridge inspection records from the National Bridge Inventory to predict whether a bridge's reported `LOWEST_RATING` is likely to decrease in the next annual data release.

The output is not a structural safety decision. The model only creates a **priority list** that can help analysts and engineers decide which bridge records deserve earlier review.

## 2. Portfolio story: Career Experience Practicum with IHI

This project is framed as a second important portfolio project based on my **Career Experience Practicum course with IHI**.

During the practicum, I worked on infrastructure and bridge related business thinking. This prototype extends that experience into a data automation idea:

**How could an infrastructure company like IHI automate the early screening of bridge inspection data?**

A manual workflow would require analysts to compare many annual bridge records, rating changes, traffic indicators, age, and condition fields. This prototype automates that first screening step by turning historical inspection data into a machine learning based risk ranking.

## 3. Automation task idea for IHI

### Manual task

Analysts or engineers need to check bridge inspection records year by year and identify which bridges may need closer review.

### Automated task

The system automatically:

1. Loads bridge inspection data from multiple years
2. Matches the same bridge across consecutive years
3. Creates a target variable: whether the next year rating decreased
4. Trains machine learning models
5. Evaluates the model using a temporal split
6. Scores the latest bridge records
7. Produces a ranked priority list for review
8. Displays the result in a dashboard prototype

### Business value

The system does not replace engineers. It helps the team start from a smaller, higher priority group instead of reviewing all bridge records with the same level of attention.

## 4. Dataset

The project uses California annual National Bridge Inventory files:

| Year | File |
|---|---|
| 2021 | `CA21.txt` |
| 2022 | `CA22.txt` |
| 2023 | `CA23.txt` |
| 2024 | `CA24.txt` |
| 2025 | `CA25.txt` |

The raw files are stored in:

```text
data/raw/
```

The processed transition dataset is stored in:

```text
data/processed/california_bridge_transitions_2021_2025.csv.gz
```

## 5. Machine learning problem

### Input

Bridge attributes from the current year, such as:

| Feature group | Examples |
|---|---|
| Age and reconstruction | `bridge_age`, `years_since_reconstruction` |
| Traffic | `ADT_029`, `PERCENT_ADT_TRUCK_109` |
| Geometry | `STRUCTURE_LEN_MT_049`, `DECK_WIDTH_MT_052` |
| Condition | `DECK_COND_NUM`, `SUPERSTRUCTURE_COND_NUM`, `LOWEST_RATING` |
| Type and ownership | `STRUCTURE_KIND_043A`, `OWNER_022`, `DESIGN_LOAD_031` |

### Target

```text
deteriorated_next_year = 1
```

when a bridge's `LOWEST_RATING` is lower in the next annual file.

```text
deteriorated_next_year = 0
```

when the rating stays the same or improves.

## 6. Validation design

This project uses a temporal split because bridge risk prediction should be tested on future data, not randomly mixed records.

| Split | Transition |
|---|---|
| Train | 2021 to 2022 and 2022 to 2023 |
| Validation | 2023 to 2024 |
| Test | 2024 to 2025 |

This is more realistic than random split because the final evaluation uses a later period the model did not see during training.

## 7. Models

The prototype compares:

| Model | Role |
|---|---|
| Logistic Regression | Baseline model, simple and explainable |
| Decision Tree | Lightweight nonlinear model with feature importance |

The best model is selected by validation PR AUC because deterioration is a rare event and accuracy would be misleading.

## 8. Current test result

The current generated report selected:

```text
Logistic Regression
```

Main test metrics from `reports/test_metrics.json`:

| Metric | Value |
|---|---:|
| ROC AUC | 0.7197 |
| PR AUC | 0.0879 |
| Precision | 0.0900 |
| Recall | 0.3586 |
| F1 | 0.1438 |

Interpretation: the model has useful ranking signal, but it should be treated as an early screening tool. It is not accurate enough to make engineering decisions alone.

## 9. Ranking value

The report `reports/top_k_ranking_metrics.csv` measures whether the highest risk group contains more actual deterioration cases than random selection.

Example from the current run:

| Priority group | Lift vs random |
|---|---:|
| Top 1 percent | about 3.80x |
| Top 5 percent | about 2.93x |
| Top 10 percent | about 2.63x |

This is useful for presentation because it shows the model can prioritize review, even though the absolute deterioration rate is low.

## 10. Project structure

```text
california_bridge_deterioration/
│
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── raw/
│   │   ├── CA21.txt
│   │   ├── CA22.txt
│   │   ├── CA23.txt
│   │   ├── CA24.txt
│   │   └── CA25.txt
│   └── processed/
│       └── california_bridge_transitions_2021_2025.csv.gz
│
├── src/
│   └── bridge_risk_pipeline.py
│
├── outputs/
│   ├── california_bridge_2026_priority_list.csv
│   └── top_100_bridge_priorities.csv
│
├── reports/
│   ├── test_metrics.json
│   ├── validation_model_comparison.csv
│   ├── target_summary.csv
│   ├── top_k_ranking_metrics.csv
│   └── feature_importance.csv
│
├── models/
│   └── bridge_deterioration_model.joblib
│
├── notebooks/
│   └── 01_bridge_risk_prototype.ipynb
│
└── docs/
    └── presentation_talking_points.md
```

## 11. How to run

### Step 1: Create virtual environment

```bash
python -m venv .venv
```

### Step 2: Activate virtual environment on Windows Git Bash

```bash
source .venv/Scripts/activate
```

### Step 3: Install packages

```bash
pip install -r requirements.txt
```

### Step 4: Run the full pipeline

```bash
python src/bridge_risk_pipeline.py --fast
```

This creates the files in `outputs/`, `reports/`, and `models/`.

### Step 5: Run the dashboard prototype

```bash
streamlit run app.py
```


## 13. Safety limitation

This project forecasts changes in reported bridge condition ratings. It does not predict collapse, certify structural safety, or replace qualified inspection and engineering review.

## 14. Future work

Possible improvements:

1. Add maintenance history and inspection notes
2. Add climate, flood, seismic, and traffic trend data
3. Calibrate risk probability with more historical years
4. Ask bridge engineers to validate the target definition
5. Deploy the dashboard on a secure internal cloud environment
6. Add model monitoring and data quality checks
