"""California Bridge Deterioration Risk Prototype.

Bối cảnh portfolio:
Project này mô phỏng một ý tưởng tự động hóa cho Career Experience Practicum với IHI.
Thay vì kiểm tra dữ liệu cầu thủ công qua nhiều file hằng năm, pipeline này tự động:
1. đọc dữ liệu National Bridge Inventory của California,
2. tạo nhãn xuống cấp rating năm sau,
3. train mô hình machine learning,
4. tạo danh sách cầu ưu tiên để kỹ sư xem xét trước.

Giới hạn an toàn:
Đây là risk ranking prototype, không phải công cụ chứng nhận an toàn cầu.
Mọi quyết định kỹ thuật vẫn cần kỹ sư cầu đường kiểm tra.
"""

# argparse giúp người dùng chạy pipeline từ terminal bằng tham số như --fast.
import argparse

# json dùng để lưu metrics thành file test_metrics.json.
import json

# warnings dùng để ẩn warning không quan trọng khi present.
import warnings

# Path giúp xử lý đường dẫn ổn định trên Windows, Mac và Linux.
from pathlib import Path

# joblib dùng để lưu mô hình đã train.
import joblib

# numpy dùng để tính toán số học và xử lý mảng xác suất.
import numpy as np

# pandas dùng để đọc, làm sạch, ghép và lưu dữ liệu dạng bảng.
import pandas as pd

# clone tạo bản sao model sạch trước khi train nếu cần.
from sklearn.base import clone

# ColumnTransformer cho phép xử lý cột số và cột phân loại theo cách khác nhau.
from sklearn.compose import ColumnTransformer

# Logistic Regression là baseline model dễ giải thích.
from sklearn.linear_model import LogisticRegression

# Decision Tree là model phi tuyến nhẹ, nhanh, và có feature importance.
from sklearn.tree import DecisionTreeClassifier

# Các metric dùng để đánh giá bài toán rare event.
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

# Pipeline giúp ghép preprocessing và model thành một quy trình thống nhất.
from sklearn.pipeline import Pipeline

# SimpleImputer xử lý missing values, OneHotEncoder xử lý categorical variables.
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Tắt warning để output dễ nhìn hơn khi demo.
warnings.filterwarnings("ignore")

# Giữ random seed cố định để kết quả ổn định khi chạy lại.
RANDOM_STATE = 42

# Năm dữ liệu đang dùng trong prototype.
YEARS = [2021, 2022, 2023, 2024, 2025]

# Những cột cần đọc từ file NBI gốc. Không đọc hết để giảm RAM và thời gian chạy.
RAW_COLUMNS = [
    "STRUCTURE_NUMBER_008", "YEAR_BUILT_027", "YEAR_RECONSTRUCTED_106",
    "ADT_029", "PERCENT_ADT_TRUCK_109", "TRAFFIC_LANES_ON_028A",
    "MAIN_UNIT_SPANS_045", "APPR_SPANS_046", "MAX_SPAN_LEN_MT_048",
    "STRUCTURE_LEN_MT_049", "ROADWAY_WIDTH_MT_051", "DECK_WIDTH_MT_052",
    "DETOUR_KILOS_019", "OPERATING_RATING_064", "INVENTORY_RATING_066",
    "INSPECT_FREQ_MONTHS_091", "DECK_COND_058", "SUPERSTRUCTURE_COND_059",
    "SUBSTRUCTURE_COND_060", "CULVERT_COND_062", "SCOUR_CRITICAL_113",
    "STRUCTURE_KIND_043A", "STRUCTURE_TYPE_043B", "DECK_STRUCTURE_TYPE_107",
    "OWNER_022", "MAINTENANCE_021", "FUNCTIONAL_CLASS_026", "SERVICE_ON_042A",
    "SERVICE_UND_042B", "DESIGN_LOAD_031", "HIGHWAY_SYSTEM_104",
    "LOWEST_RATING", "BRIDGE_CONDITION", "LAT_016", "LONG_017",
    "FACILITY_CARRIED_007", "LOCATION_009",
]

# Biến số dùng để train model.
NUMERIC_FEATURES = [
    "bridge_age", "years_since_reconstruction", "ADT_029", "PERCENT_ADT_TRUCK_109",
    "TRAFFIC_LANES_ON_028A", "MAIN_UNIT_SPANS_045", "APPR_SPANS_046",
    "MAX_SPAN_LEN_MT_048", "STRUCTURE_LEN_MT_049", "ROADWAY_WIDTH_MT_051",
    "DECK_WIDTH_MT_052", "DETOUR_KILOS_019", "OPERATING_RATING_064",
    "INVENTORY_RATING_066", "INSPECT_FREQ_MONTHS_091", "DECK_COND_NUM",
    "SUPERSTRUCTURE_COND_NUM", "SUBSTRUCTURE_COND_NUM", "CULVERT_COND_NUM",
    "LOWEST_RATING",
]

# Biến phân loại dùng để train model.
CATEGORICAL_FEATURES = [
    "SCOUR_CRITICAL_113", "STRUCTURE_KIND_043A", "STRUCTURE_TYPE_043B",
    "DECK_STRUCTURE_TYPE_107", "OWNER_022", "MAINTENANCE_021",
    "FUNCTIONAL_CLASS_026", "SERVICE_ON_042A", "SERVICE_UND_042B",
    "DESIGN_LOAD_031", "HIGHWAY_SYSTEM_104", "BRIDGE_CONDITION",
]

# Tổng hợp toàn bộ feature đầu vào của model.
MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def project_root() -> Path:
    """Lấy thư mục gốc của project.

    Hàm này giúp code không phụ thuộc vào việc bạn chạy từ VS Code hay terminal.
    """
    # __file__ là vị trí file bridge_risk_pipeline.py.
    this_file = Path(__file__).resolve()
    # File nằm trong src, nên parents[1] chính là thư mục project.
    return this_file.parents[1]


def nbi_coordinate_to_decimal(value: object, is_longitude: bool = False) -> float:
    """Chuyển tọa độ NBI từ độ phút giây sang decimal degree.

    Plotly map cần latitude và longitude dạng số thập phân.
    File NBI lại lưu theo dạng DDMMSSss hoặc DDDMMSSss.
    """
    # Ép giá trị sang số, lỗi thì thành NaN.
    number = pd.to_numeric(value, errors="coerce")
    # Nếu bị thiếu dữ liệu, trả về NaN.
    if pd.isna(number):
        return np.nan
    # Kinh độ có 3 chữ số độ, vĩ độ có 2 chữ số độ.
    degree_digits = 3 if is_longitude else 2
    # Tổng độ dài chuẩn là phần độ cộng 6 chữ số phút và giây.
    total_digits = degree_digits + 6
    # Zfill giúp tránh lỗi khi số bị thiếu chữ số 0 ở đầu.
    text = str(int(round(abs(float(number))))).zfill(total_digits)
    # Tách phần độ.
    degrees = int(text[:degree_digits])
    # Tách phần phút.
    minutes = int(text[degree_digits:degree_digits + 2])
    # Tách phần giây, chia 100 vì NBI lưu giây với 2 chữ số thập phân.
    seconds = int(text[degree_digits + 2:]) / 100
    # Đổi độ phút giây thành số thập phân.
    decimal = degrees + minutes / 60 + seconds / 3600
    # California ở Tây bán cầu nên longitude là số âm.
    return -decimal if is_longitude else decimal


def read_nbi_file(path: Path, year: int) -> pd.DataFrame:
    """Đọc một file NBI của một năm."""
    # Đọc file txt, chỉ lấy các cột cần thiết.
    frame = pd.read_csv(path, usecols=RAW_COLUMNS, quotechar="'", low_memory=False)
    # Chuẩn hóa mã cầu để ghép cùng một cây cầu qua nhiều năm.
    frame["bridge_id"] = frame["STRUCTURE_NUMBER_008"].astype(str).str.strip()
    # Thêm năm để tracking nguồn dữ liệu.
    frame["year"] = year
    # Trả về dataframe đã đọc.
    return frame


def load_raw_frames(raw_dir: Path) -> dict[int, pd.DataFrame]:
    """Đọc các file CA21.txt đến CA25.txt vào dictionary."""
    # Tạo dictionary rỗng để lưu dataframe theo năm.
    frames: dict[int, pd.DataFrame] = {}
    # Lặp qua từng năm trong danh sách YEARS.
    for year in YEARS:
        # Lấy 2 chữ số cuối của năm, ví dụ 2021 thành 21.
        short_year = str(year)[-2:]
        # Tạo đường dẫn file raw.
        path = raw_dir / f"CA{short_year}.txt"
        # Nếu file không tồn tại, báo lỗi rõ ràng.
        if not path.exists():
            raise FileNotFoundError(f"Missing raw data file: {path}")
        # Đọc file và lưu vào dictionary.
        frames[year] = read_nbi_file(path, year)
    # Trả về toàn bộ dữ liệu gốc.
    return frames


def clean_current_year(frame: pd.DataFrame, year: int) -> pd.DataFrame:
    """Làm sạch dữ liệu một năm và tạo feature cho model."""
    # Tạo dataframe mới để không sửa dữ liệu gốc.
    out = pd.DataFrame(index=frame.index)
    # Giữ bridge_id để merge qua năm sau.
    out["bridge_id"] = frame["bridge_id"]
    # Giữ năm hiện tại.
    out["year"] = year
    # Ép năm xây dựng sang số.
    year_built = pd.to_numeric(frame["YEAR_BUILT_027"], errors="coerce")
    # Ép năm tái xây dựng sang số.
    year_reconstructed = pd.to_numeric(frame["YEAR_RECONSTRUCTED_106"], errors="coerce")
    # Tính tuổi cầu.
    out["bridge_age"] = year - year_built
    # Tính số năm từ lần tái xây dựng, nếu không có thì để NaN.
    out["years_since_reconstruction"] = np.where(year_reconstructed > 0, year - year_reconstructed, np.nan)
    # Các cột số cần ép kiểu.
    numeric_columns = [
        "ADT_029", "PERCENT_ADT_TRUCK_109", "TRAFFIC_LANES_ON_028A",
        "MAIN_UNIT_SPANS_045", "APPR_SPANS_046", "MAX_SPAN_LEN_MT_048",
        "STRUCTURE_LEN_MT_049", "ROADWAY_WIDTH_MT_051", "DECK_WIDTH_MT_052",
        "DETOUR_KILOS_019", "OPERATING_RATING_064", "INVENTORY_RATING_066",
        "INSPECT_FREQ_MONTHS_091", "LOWEST_RATING", "LAT_016", "LONG_017",
    ]
    # Lặp từng cột số và chuyển về numeric.
    for column in numeric_columns:
        out[column] = pd.to_numeric(frame[column], errors="coerce")
    # Tạo latitude dạng decimal để vẽ map.
    out["latitude"] = out["LAT_016"].apply(nbi_coordinate_to_decimal)
    # Tạo longitude dạng decimal để vẽ map.
    out["longitude"] = out["LONG_017"].apply(lambda value: nbi_coordinate_to_decimal(value, is_longitude=True))
    # Mapping cột condition gốc sang tên dễ hiểu.
    condition_columns = {
        "DECK_COND_058": "DECK_COND_NUM",
        "SUPERSTRUCTURE_COND_059": "SUPERSTRUCTURE_COND_NUM",
        "SUBSTRUCTURE_COND_060": "SUBSTRUCTURE_COND_NUM",
        "CULVERT_COND_062": "CULVERT_COND_NUM",
    }
    # Chuyển từng rating condition sang số.
    for source, destination in condition_columns.items():
        out[destination] = pd.to_numeric(frame[source], errors="coerce")
    # Làm sạch biến phân loại.
    for column in CATEGORICAL_FEATURES:
        out[column] = frame[column].astype("string").fillna("Missing")
    # Giữ tên tuyến hoặc công trình cầu phục vụ để hiển thị.
    out["FACILITY_CARRIED_007"] = frame["FACILITY_CARRIED_007"].astype("string").fillna("")
    # Giữ mô tả vị trí để hiển thị trong dashboard.
    out["LOCATION_009"] = frame["LOCATION_009"].astype("string").fillna("")
    # Trả về dữ liệu đã sạch.
    return out


def make_transition(raw_frames: dict[int, pd.DataFrame], current_year: int) -> pd.DataFrame:
    """Tạo dataset dự đoán từ một cặp năm liên tiếp."""
    # Làm sạch dữ liệu năm hiện tại.
    current = clean_current_year(raw_frames[current_year], current_year)
    # Lấy rating năm sau để tạo target.
    following = raw_frames[current_year + 1][["bridge_id", "LOWEST_RATING", "BRIDGE_CONDITION"]].copy()
    # Đổi tên cột năm sau để tránh nhầm với rating năm hiện tại.
    following = following.rename(columns={"LOWEST_RATING": "next_lowest_rating", "BRIDGE_CONDITION": "next_bridge_condition"})
    # Ép rating năm sau sang số.
    following["next_lowest_rating"] = pd.to_numeric(following["next_lowest_rating"], errors="coerce")
    # Ghép cùng một cây cầu giữa năm hiện tại và năm sau.
    paired = current.merge(following, on="bridge_id", how="inner")
    # Target bằng 1 nếu rating năm sau thấp hơn rating năm hiện tại.
    paired["deteriorated_next_year"] = (paired["next_lowest_rating"] < paired["LOWEST_RATING"]).astype("int8")
    # Ghi nhãn transition để chia train validation test theo thời gian.
    paired["transition"] = f"{current_year}_to_{current_year + 1}"
    # Trả về dataset transition.
    return paired


def build_transition_dataset(raw_frames: dict[int, pd.DataFrame]) -> pd.DataFrame:
    """Gộp các transition 2021→2022 đến 2024→2025."""
    # Tạo list transition cho 4 cặp năm liên tiếp.
    transition_list = [make_transition(raw_frames, year) for year in range(2021, 2025)]
    # Gộp các bảng thành một dataset lớn.
    return pd.concat(transition_list, ignore_index=True)


def split_temporally(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chia dữ liệu theo thời gian thay vì random split."""
    # Train dùng 2 transition cũ nhất.
    train = data[data["transition"].isin(["2021_to_2022", "2022_to_2023"])].copy()
    # Validation dùng transition 2023→2024.
    validation = data[data["transition"] == "2023_to_2024"].copy()
    # Test dùng transition mới nhất 2024→2025.
    test = data[data["transition"] == "2024_to_2025"].copy()
    # Trả về 3 tập dữ liệu.
    return train, validation, test


def make_preprocessor() -> ColumnTransformer:
    """Tạo preprocessing pipeline cho cột số và cột phân loại."""
    # Cột số được điền median và scale.
    numeric_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    # Cột phân loại được điền most frequent và one hot encoding.
    categorical_pipeline = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=20))])
    # ColumnTransformer áp dụng pipeline đúng cho từng nhóm cột.
    return ColumnTransformer([("numeric", numeric_pipeline, NUMERIC_FEATURES), ("categorical", categorical_pipeline, CATEGORICAL_FEATURES)])


def make_models(fast: bool = False) -> dict[str, Pipeline]:
    """Tạo hai model để so sánh."""
    # Lấy preprocessing chung cho các model.
    preprocessor = make_preprocessor()
    # Logistic Regression là baseline mạnh cho dữ liệu bảng và dễ explain.
    logistic = Pipeline([("preprocessor", clone(preprocessor)), ("classifier", LogisticRegression(max_iter=400, solver="liblinear", class_weight="balanced", random_state=RANDOM_STATE))])
    # Decision Tree giúp bắt quan hệ phi tuyến và có feature importance.
    tree_depth = 6 if fast else 8
    # Tạo Decision Tree với giới hạn độ sâu để giảm overfitting.
    tree = Pipeline([("preprocessor", clone(preprocessor)), ("classifier", DecisionTreeClassifier(max_depth=tree_depth, min_samples_leaf=80, class_weight="balanced", random_state=RANDOM_STATE))])
    # Trả về dictionary model.
    return {"Logistic Regression": logistic, "Decision Tree": tree}


def evaluate_probability(y_true: pd.Series, probability: np.ndarray) -> dict[str, float]:
    """Tính ROC AUC và PR AUC từ xác suất dự đoán."""
    # ROC AUC đo khả năng xếp hạng tổng quát.
    roc_auc = roc_auc_score(y_true, probability)
    # PR AUC phù hợp với bài toán rare event hơn accuracy.
    pr_auc = average_precision_score(y_true, probability)
    # Trả metrics dạng dictionary.
    return {"roc_auc": float(roc_auc), "pr_auc": float(pr_auc)}


def choose_threshold(y_true: pd.Series, probability: np.ndarray) -> tuple[float, dict[str, float]]:
    """Chọn threshold tốt nhất theo F1 trên validation set."""
    # Tạo precision recall curve.
    precision, recall, thresholds = precision_recall_curve(y_true, probability)
    # Tính F1 cho từng threshold.
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    # Lấy index có F1 cao nhất, bỏ điểm cuối vì không có threshold tương ứng.
    best_index = int(np.nanargmax(f1[:-1]))
    # Lấy threshold được chọn.
    threshold = float(thresholds[best_index])
    # Lưu thông tin validation tại threshold này.
    summary = {"validation_precision": float(precision[best_index]), "validation_recall": float(recall[best_index]), "validation_f1": float(f1[best_index])}
    # Trả threshold và summary.
    return threshold, summary


def top_k_metrics(y_true: pd.Series, probability: np.ndarray, fraction: float) -> dict[str, float]:
    """Đo hiệu quả khi chỉ review top K phần trăm cầu rủi ro cao nhất."""
    # Chuyển target sang numpy array.
    y_array = np.asarray(y_true)
    # Tính số cầu được chọn.
    count = max(1, int(len(y_array) * fraction))
    # Sắp xếp probability giảm dần và lấy top K index.
    selected = np.argsort(probability)[::-1][:count]
    # Precision at K là tỷ lệ cầu xuống cấp thật trong nhóm được ưu tiên.
    precision_at_k = float(y_array[selected].mean())
    # Recall at K là tỷ lệ case xuống cấp thật được bắt trong nhóm ưu tiên.
    recall_at_k = float(y_array[selected].sum() / max(1, y_array.sum()))
    # Lift cho biết ranking tốt hơn chọn ngẫu nhiên bao nhiêu lần.
    lift = float(precision_at_k / max(1e-12, y_array.mean()))
    # Trả kết quả dạng dictionary.
    return {"priority_fraction": float(fraction), "bridges_selected": int(count), "precision_at_k": precision_at_k, "recall_at_k": recall_at_k, "lift_vs_random": lift}


def train_and_evaluate(data: pd.DataFrame, fast: bool = False) -> dict[str, object]:
    """Train model, chọn model tốt nhất và đánh giá trên test set."""
    # Chia dữ liệu theo thời gian.
    train, validation, test = split_temporally(data)
    # Tách X và y cho train.
    X_train, y_train = train[MODEL_FEATURES], train["deteriorated_next_year"]
    # Tách X và y cho validation.
    X_validation, y_validation = validation[MODEL_FEATURES], validation["deteriorated_next_year"]
    # Tách X và y cho test.
    X_test, y_test = test[MODEL_FEATURES], test["deteriorated_next_year"]
    # Tạo model candidates.
    models = make_models(fast=fast)
    # Tạo list lưu kết quả validation.
    rows = []
    # Train và validate từng model.
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        probability = model.predict_proba(X_validation)[:, 1]
        rows.append({"model": name, **evaluate_probability(y_validation, probability)})
    # Xếp model theo PR AUC vì target là rare event.
    validation_results = pd.DataFrame(rows).sort_values("pr_auc", ascending=False).reset_index(drop=True)
    # Chọn model tốt nhất.
    best_name = str(validation_results.iloc[0]["model"])
    # Lấy object model tốt nhất.
    best_model = models[best_name]
    # Tính probability trên validation để chọn threshold.
    validation_probability = best_model.predict_proba(X_validation)[:, 1]
    # Chọn threshold bằng validation F1.
    threshold, threshold_summary = choose_threshold(y_validation, validation_probability)
    # Tính probability trên test set.
    test_probability = best_model.predict_proba(X_test)[:, 1]
    # Đổi probability thành nhãn dự đoán.
    test_prediction = (test_probability >= threshold).astype(int)
    # Tính metrics cuối cùng trên test.
    test_metrics = {"model": best_name, "roc_auc": float(roc_auc_score(y_test, test_probability)), "pr_auc": float(average_precision_score(y_test, test_probability)), "precision": float(precision_score(y_test, test_prediction, zero_division=0)), "recall": float(recall_score(y_test, test_prediction, zero_division=0)), "f1": float(f1_score(y_test, test_prediction, zero_division=0)), "threshold": float(threshold), **threshold_summary}
    # Tạo bảng top K ranking.
    ranking = pd.DataFrame([top_k_metrics(y_test, test_probability, f) for f in [0.01, 0.05, 0.10, 0.20]])
    # Trả toàn bộ kết quả cần dùng sau.
    return {"models": models, "best_model": best_model, "best_model_name": best_name, "validation_results": validation_results, "test_metrics": test_metrics, "ranking_results": ranking}


def create_priority_list(raw_frames: dict[int, pd.DataFrame], best_model: Pipeline) -> pd.DataFrame:
    """Score dữ liệu cầu mới nhất năm 2025 để tạo priority list."""
    # Làm sạch dữ liệu 2025.
    latest = clean_current_year(raw_frames[2025], 2025)
    # Dự đoán xác suất rating giảm ở năm tiếp theo.
    probability = best_model.predict_proba(latest[MODEL_FEATURES])[:, 1]
    # Chọn các cột cần cho dashboard.
    output = latest[["bridge_id", "FACILITY_CARRIED_007", "LOCATION_009", "latitude", "longitude", "LOWEST_RATING", "BRIDGE_CONDITION", "bridge_age", "ADT_029", "STRUCTURE_LEN_MT_049", "DECK_COND_NUM", "SUPERSTRUCTURE_COND_NUM", "SUBSTRUCTURE_COND_NUM"]].copy()
    # Gắn probability vào output.
    output["predicted_deterioration_probability"] = probability
    # Tạo rank, rank 1 là cầu có score cao nhất.
    output["priority_rank"] = output["predicted_deterioration_probability"].rank(method="first", ascending=False).astype(int)
    # Sắp xếp theo priority rank.
    output = output.sort_values("priority_rank").reset_index(drop=True)
    # Trả về priority list.
    return output


def save_feature_importance(model: Pipeline, path: Path) -> None:
    """Lưu feature importance nếu model có hỗ trợ."""
    # Lấy classifier từ pipeline.
    classifier = model.named_steps["classifier"]
    # Nếu classifier không có feature_importances_, tạo file rỗng.
    if not hasattr(classifier, "feature_importances_"):
        pd.DataFrame(columns=["feature", "importance"]).to_csv(path, index=False)
        return
    # Lấy tên feature sau preprocessing.
    names = model.named_steps["preprocessor"].get_feature_names_out()
    # Tạo bảng feature importance.
    importance = pd.DataFrame({"feature": names, "importance": classifier.feature_importances_}).sort_values("importance", ascending=False)
    # Lưu bảng ra CSV.
    importance.to_csv(path, index=False)


def save_outputs(root: Path, data: pd.DataFrame, priority: pd.DataFrame, results: dict[str, object]) -> None:
    """Lưu toàn bộ file output cho dashboard và GitHub."""
    # Tạo các thư mục cần thiết.
    (root / "outputs").mkdir(exist_ok=True)
    (root / "reports").mkdir(exist_ok=True)
    (root / "models").mkdir(exist_ok=True)
    (root / "data" / "processed").mkdir(parents=True, exist_ok=True)
    # Lưu transition dataset đã xử lý.
    data.to_csv(root / "data" / "processed" / "california_bridge_transitions_2021_2025.csv.gz", index=False)
    # Lưu toàn bộ priority list.
    priority.to_csv(root / "outputs" / "california_bridge_2026_priority_list.csv", index=False)
    # Lưu top 100 để xem nhanh trên GitHub.
    priority.head(100).to_csv(root / "outputs" / "top_100_bridge_priorities.csv", index=False)
    # Lưu model comparison.
    results["validation_results"].to_csv(root / "reports" / "validation_model_comparison.csv", index=False)
    # Lưu ranking metrics.
    results["ranking_results"].to_csv(root / "reports" / "top_k_ranking_metrics.csv", index=False)
    # Lưu test metrics.
    (root / "reports" / "test_metrics.json").write_text(json.dumps(results["test_metrics"], indent=2), encoding="utf-8")
    # Lưu thống kê target theo transition.
    target_summary = data.groupby("transition")["deteriorated_next_year"].agg(["count", "sum", "mean"]).rename(columns={"count": "bridges", "sum": "deteriorated_bridges", "mean": "deterioration_rate"}).reset_index()
    target_summary.to_csv(root / "reports" / "target_summary.csv", index=False)
    # Lưu feature importance từ Decision Tree để giải thích mô hình.
    save_feature_importance(results["models"]["Decision Tree"], root / "reports" / "feature_importance.csv")
    # Lưu best model đã train.
    joblib.dump(results["best_model"], root / "models" / "bridge_deterioration_model.joblib")


def run_pipeline(root: Path | None = None, fast: bool = False) -> dict[str, object]:
    """Chạy toàn bộ pipeline từ raw data đến prototype output."""
    # Nếu không truyền root, tự tìm project root.
    root = root or project_root()
    # Đọc raw files.
    raw_frames = load_raw_frames(root / "data" / "raw")
    # Tạo transition dataset.
    transition_data = build_transition_dataset(raw_frames)
    # Train và đánh giá model.
    results = train_and_evaluate(transition_data, fast=fast)
    # Tạo priority list cho dữ liệu 2025.
    priority = create_priority_list(raw_frames, results["best_model"])
    # Lưu output cho dashboard.
    save_outputs(root, transition_data, priority, results)
    # In thông báo hoàn thành.
    print("Pipeline completed successfully.")
    print("Best model:", results["best_model_name"])
    print("Output file: outputs/california_bridge_2026_priority_list.csv")
    # Trả kết quả để notebook có thể dùng tiếp.
    return {"transition_data": transition_data, "priority": priority, **results}


def main() -> None:
    """Hàm chính khi chạy bằng terminal."""
    # Tạo parser cho command line.
    parser = argparse.ArgumentParser(description="Run bridge deterioration risk pipeline")
    # Thêm tham số --fast để demo nhanh hơn.
    parser.add_argument("--fast", action="store_true", help="Run with a smaller Decision Tree")
    # Đọc tham số người dùng nhập.
    args = parser.parse_args()
    # Chạy pipeline.
    run_pipeline(fast=args.fast)


# Chỉ chạy main nếu file được gọi trực tiếp bằng python.
if __name__ == "__main__":
    main()
