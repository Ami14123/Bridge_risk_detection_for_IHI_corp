"""Streamlit dashboard cho Bridge Risk Intelligence Prototype.

Chạy bằng:
streamlit run app.py
"""

# Path giúp tạo đường dẫn file ổn định.
from pathlib import Path

# json dùng để đọc test metrics.
import json

# pandas dùng để đọc bảng CSV.
import pandas as pd

# plotly express dùng để vẽ chart và map tương tác.
import plotly.express as px

# streamlit dùng để dựng web prototype nhanh.
import streamlit as st

# PROJECT_ROOT là thư mục chứa app.py.
PROJECT_ROOT = Path(__file__).resolve().parent

# File prediction đã được tạo bởi pipeline.
PREDICTION_FILE = PROJECT_ROOT / "outputs" / "california_bridge_2026_priority_list.csv"

# File metrics đã được tạo bởi pipeline.
METRICS_FILE = PROJECT_ROOT / "reports" / "test_metrics.json"

# File thống kê target theo từng transition.
TARGET_FILE = PROJECT_ROOT / "reports" / "target_summary.csv"

# File đánh giá top K ranking.
RANKING_FILE = PROJECT_ROOT / "reports" / "top_k_ranking_metrics.csv"


def load_predictions() -> pd.DataFrame:
    """Đọc danh sách cầu ưu tiên.

    App chỉ đọc file output, không train lại model, nên chạy rất nhanh khi present.
    """
    # Nếu chưa có file output, hướng dẫn chạy pipeline trước.
    if not PREDICTION_FILE.exists():
        st.error("Chưa có prediction file. Hãy chạy: python src/bridge_risk_pipeline.py --fast")
        st.stop()
    # Đọc file CSV thành dataframe.
    return pd.read_csv(PREDICTION_FILE)


def load_json(path: Path) -> dict:
    """Đọc file JSON, nếu thiếu thì trả dictionary rỗng."""
    # Nếu file không tồn tại, trả về rỗng để app không lỗi.
    if not path.exists():
        return {}
    # Đọc nội dung JSON.
    return json.loads(path.read_text(encoding="utf-8"))


def load_csv(path: Path) -> pd.DataFrame:
    """Đọc file CSV report, nếu thiếu thì trả dataframe rỗng."""
    # Nếu file không tồn tại, trả dataframe rỗng.
    if not path.exists():
        return pd.DataFrame()
    # Đọc CSV bằng pandas.
    return pd.read_csv(path)


def show_header() -> None:
    """Hiển thị phần giới thiệu business của prototype."""
    # Tiêu đề chính của dashboard.
    st.title("Bridge Risk Intelligence Prototype")
    # Mô tả ngắn gọn gắn với IHI Career Experience Practicum.
    st.write(
        "Prototype này mô phỏng cách tự động hóa một bước phân tích trong bối cảnh Career Experience Practicum với IHI: "
        "từ dữ liệu kiểm định cầu hằng năm, hệ thống tự động tạo risk score và danh sách cầu nên được xem xét trước."
    )
    # Cảnh báo giới hạn an toàn.
    st.warning("Đây là công cụ ưu tiên review, không phải công cụ chứng nhận an toàn cầu.")


def show_metrics(metrics: dict, predictions: pd.DataFrame) -> None:
    """Hiển thị 4 metric chính ở đầu dashboard."""
    # Tạo 4 cột ngang.
    col1, col2, col3, col4 = st.columns(4)
    # Hiển thị model được chọn.
    col1.metric("Selected model", metrics.get("model", "N/A"))
    # Hiển thị PR AUC vì target là rare event.
    col2.metric("Test PR AUC", round(metrics.get("pr_auc", 0), 4))
    # Hiển thị recall trên test set.
    col3.metric("Test recall", round(metrics.get("recall", 0), 4))
    # Hiển thị số cầu được score.
    col4.metric("Scored bridges", f"{len(predictions):,}")


def show_priority_dashboard(predictions: pd.DataFrame) -> None:
    """Hiển thị map và bảng top bridge priorities."""
    # Slider cho phép chọn số cầu muốn xem.
    top_n = st.slider("Số cầu ưu tiên muốn xem", 10, min(1000, len(predictions)), 100)
    # Lấy top N cầu rủi ro cao nhất.
    top_data = predictions.head(top_n).copy()
    # Lọc những dòng có tọa độ để vẽ map.
    map_data = top_data.dropna(subset=["latitude", "longitude"])
    # Vẽ map nếu có dữ liệu tọa độ.
    if not map_data.empty:
        fig = px.scatter_map(
            map_data,
            lat="latitude",
            lon="longitude",
            color="predicted_deterioration_probability",
            hover_name="bridge_id",
            hover_data=["FACILITY_CARRIED_007", "LOCATION_009", "LOWEST_RATING", "BRIDGE_CONDITION", "priority_rank"],
            zoom=4.5,
            height=600,
            title="Top bridge priorities on map",
        )
        st.plotly_chart(fig, use_container_width=True)
    # Chọn cột chính cho bảng.
    table_columns = ["priority_rank", "bridge_id", "FACILITY_CARRIED_007", "LOCATION_009", "predicted_deterioration_probability", "LOWEST_RATING", "BRIDGE_CONDITION", "bridge_age", "ADT_029"]
    # Hiển thị bảng top N.
    st.dataframe(top_data[table_columns], use_container_width=True, hide_index=True)


def show_model_report(metrics: dict) -> None:
    """Hiển thị báo cáo model và ranking value."""
    # Đọc target summary.
    target = load_csv(TARGET_FILE)
    # Nếu có target summary, vẽ bar chart.
    if not target.empty:
        fig = px.bar(target, x="transition", y="deterioration_rate", title="Observed deterioration rate by transition")
        st.plotly_chart(fig, use_container_width=True)
    # Đọc top K ranking report.
    ranking = load_csv(RANKING_FILE)
    # Nếu có ranking report, hiển thị bảng.
    if not ranking.empty:
        st.subheader("Top priority ranking value")
        st.dataframe(ranking, use_container_width=True, hide_index=True)
    # Hiển thị raw metrics JSON.
    st.subheader("Test metrics")
    st.json(metrics)


def show_ihi_story() -> None:
    """Hiển thị câu chuyện present gắn với IHI."""
    # Tiêu đề tab story.
    st.subheader("Automation story for IHI")
    # Mô tả workflow thủ công.
    st.write("Manual workflow: analysts compare annual bridge records, rating changes, traffic, age, and condition fields manually.")
    # Mô tả workflow tự động.
    st.write("Automated workflow: the system loads historical files, creates a deterioration target, trains models, and produces a ranked priority list.")
    # Mô tả giá trị business.
    st.write("Value: engineers can start from a smaller high priority group instead of reviewing every bridge record with the same attention level.")
    # Nhấn mạnh human in the loop.
    st.write("Human in the loop: model output is only a first screening layer. Engineers still make the final decision.")


def main() -> None:
    """Chạy toàn bộ Streamlit app."""
    # Set layout rộng cho dashboard.
    st.set_page_config(page_title="Bridge Risk Prototype", layout="wide")
    # Đọc predictions.
    predictions = load_predictions()
    # Đọc metrics.
    metrics = load_json(METRICS_FILE)
    # Hiển thị header.
    show_header()
    # Hiển thị metrics cards.
    show_metrics(metrics, predictions)
    # Tạo 3 tab cho presentation.
    tab1, tab2, tab3 = st.tabs(["Priority Dashboard", "Model Report", "IHI Automation Story"])
    # Tab dashboard.
    with tab1:
        show_priority_dashboard(predictions)
    # Tab report.
    with tab2:
        show_model_report(metrics)
    # Tab story.
    with tab3:
        show_ihi_story()


# Chỉ chạy main khi gọi file trực tiếp.
if __name__ == "__main__":
    main()
