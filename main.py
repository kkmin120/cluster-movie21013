

import math

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# ==========================================
# 1. 페이지 설정
# ==========================================
st.set_page_config(
    page_title="영화 유형 나누기",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 영화 유형 나누기")
st.write(
    "영화의 특징을 바탕으로 비슷한 영화끼리 묶어 봅니다. "
    "묶음 수와 분석 속성을 바꾸면서 결과를 비교할 수 있습니다."
)

DATA_URL = (
    "https://raw.githubusercontent.com/"
    "it-teacher-go/data-share-/main/kobis_movies.csv"
)

FEATURES = [
    "스크린 수 (로그)",
    "누적 관객 (로그)",
    "10위권 일수",
    "롱런 지수"
]

LABELS = ["㉮", "㉯", "㉰", "㉱", "㉲", "㉳", "㉴"]

COLORS = {
    "㉮": "#E76F51",
    "㉯": "#457B9D",
    "㉰": "#2A9D8F",
    "㉱": "#E9C46A",
    "㉲": "#8D6CAB",
    "㉳": "#E76FAD",
    "㉴": "#6C757D"
}


# ==========================================
# 2. 데이터 불러오기
# ==========================================
@st.cache_data
def load_data():
    data = pd.read_csv(DATA_URL, encoding="utf-8")

    required = [
        "movieCd", "movieNm", "openDt", "genre", "nation",
        "first_scrn", "first_show", "first_date", "peak",
        "first_week_audi", "total_audi", "days_in_top10"
    ]

    missing = [col for col in required if col not in data.columns]

    if missing:
        raise ValueError(
            "CSV 파일에 필요한 열이 없습니다: "
            + ", ".join(missing)
        )

    return data


try:
    original = load_data()
except Exception as error:
    st.error("영화 데이터를 불러오지 못했습니다.")
    st.code(str(error))
    st.stop()


# ==========================================
# 3. 데이터 정리
# ==========================================
total_count = len(original)
df = original.copy()

number_columns = [
    "first_scrn",
    "first_week_audi",
    "total_audi",
    "days_in_top10"
]

for col in number_columns:
    df[col] = pd.to_numeric(
        df[col].astype("string").str.replace(",", "", regex=False),
        errors="coerce"
    )

valid = (
    df[number_columns].notna().all(axis=1)
    & (df["first_scrn"] > 0)
    & (df["total_audi"] > 0)
    & (df["first_week_audi"] > 0)
)

df = df.loc[valid].copy()
df = df.reset_index(drop=True)

df["스크린 수 (로그)"] = df["first_scrn"].map(math.log10)
df["누적 관객 (로그)"] = df["total_audi"].map(math.log10)
df["10위권 일수"] = df["days_in_top10"]
df["롱런 지수"] = (
    df["total_audi"] / df["first_week_audi"]
).clip(upper=20)

st.markdown(
    f"**전체 영화: {total_count:,}편　|　"
    f"분석에 사용한 영화: {len(df):,}편**"
)

st.caption(
    "필수 속성에 값이 없거나 스크린 수, 누적 관객, 첫 주 관객이 "
    "0 이하인 영화는 제외합니다. 로그는 상용로그를 사용합니다."
)

if len(df) < 8:
    st.error("묶음 수 2~7을 비교하려면 분석할 영화가 최소 8편 필요합니다.")
    st.stop()


# ==========================================
# 4. 속성 선택과 묶음 수 선택
# ==========================================
st.subheader("1. 분석 설정")

selected = st.multiselect(
    "묶음 분석에 사용할 속성 (두 개 이상 선택)",
    options=FEATURES,
    default=FEATURES,
    key="features"
)

if len(selected) < 2:
    st.warning("속성을 두 개 이상 선택해 주세요.")
    st.stop()

chosen_k = st.selectbox(
    "영화를 몇 개의 묶음으로 나눌까요?",
    options=[2, 3, 4, 5, 6, 7],
    index=1,
    format_func=lambda k: f"{k}개 묶음",
    key="chosen_k"
)

st.caption(
    f"현재 설정: 속성 {len(selected)}개 · 묶음 {chosen_k}개"
)


# ==========================================
# 5. 표준화와 K-평균 군집 분석
# ==========================================
X = df[selected].astype(float)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 현재 선택한 묶음 수로 군집 생성
model = KMeans(
    n_clusters=chosen_k,
    random_state=42,
    n_init=10
)

df["군집 번호"] = model.fit_predict(X_scaled)

# 군집별 누적 관객 평균이 높은 순서로 기호 지정
cluster_order = (
    df.groupby("군집 번호")["total_audi"]
    .mean()
    .sort_values(ascending=False)
    .index.tolist()
)

label_map = {
    cluster_id: LABELS[i]
    for i, cluster_id in enumerate(cluster_order)
}

df["영화 유형"] = df["군집 번호"].map(label_map)

active_labels = LABELS[:chosen_k]


# ==========================================
# 6. 2차원 산점도
# ==========================================
st.subheader("2. 2차원 산점도")
st.write(
    "가로축과 세로축을 바꾸어 유형별 분포를 확인하세요. "
    "점에 마우스를 올리면 영화 제목을 볼 수 있습니다."
)

left, right = st.columns(2)

with left:
    x_axis = st.selectbox(
        "가로축 속성",
        selected,
        index=0,
        key="x_axis"
    )

y_choices = [f for f in selected if f != x_axis]

with right:
    y_axis = st.selectbox(
        "세로축 속성",
        y_choices,
        index=0,
        key="y_axis"
    )

fig2 = px.scatter(
    df,
    x=x_axis,
    y=y_axis,
    color="영화 유형",
    color_discrete_map=COLORS,
    category_orders={"영화 유형": active_labels},
    hover_name="movieNm",
    hover_data={
        "영화 유형": True,
        x_axis: ":.2f",
        y_axis: ":.2f"
    },
    labels={
        x_axis: x_axis,
        y_axis: y_axis,
        "영화 유형": "영화 유형"
    },
    title=f"{chosen_k}개 유형의 2차원 분포"
)

fig2.update_traces(
    marker={"size": 7, "opacity": 0.8}
)
fig2.update_layout(height=520)

st.plotly_chart(fig2, use_container_width=True)


# ==========================================
# 7. 3차원 산점도
# ==========================================
st.subheader("3. 3차원 산점도")

if len(selected) < 3:
    st.info(
        "3차원 산점도를 보려면 분석 속성을 세 개 이상 선택하세요."
    )
else:
    c1, c2, c3 = st.columns(3)

    with c1:
        x3 = st.selectbox(
            "X축",
            selected,
            index=0,
            key="x3"
        )

    y3_choices = [f for f in selected if f != x3]

    with c2:
        y3 = st.selectbox(
            "Y축",
            y3_choices,
            index=0,
            key="y3"
        )

    z3_choices = [
        f for f in selected
        if f != x3 and f != y3
    ]

    with c3:
        z3 = st.selectbox(
            "Z축",
            z3_choices,
            index=0,
            key="z3"
        )

    fig3 = px.scatter_3d(
        df,
        x=x3,
        y=y3,
        z=z3,
        color="영화 유형",
        color_discrete_map=COLORS,
        category_orders={"영화 유형": active_labels},
        hover_name="movieNm",
        hover_data={
            "영화 유형": True,
            x3: ":.2f",
            y3: ":.2f",
            z3: ":.2f"
        },
        labels={
            x3: x3,
            y3: y3,
            z3: z3,
            "영화 유형": "영화 유형"
        },
        title=f"{chosen_k}개 유형의 3차원 분포"
    )

    fig3.update_traces(
        marker={"size": 3, "opacity": 0.8}
    )

    fig3.update_layout(
        height=650,
        scene={
            "xaxis_title": x3,
            "yaxis_title": y3,
            "zaxis_title": z3
        }
    )

    st.plotly_chart(fig3, use_container_width=True)


# ==========================================
# 8. 유형별 평균 표
# ==========================================
st.subheader("4. 유형별 편수와 네 속성의 평균")

summary = (
    df.groupby("영화 유형")
    .agg(
        영화_편수=("movieNm", "size"),
        평균_스크린수=("first_scrn", "mean"),
        평균_누적관객=("total_audi", "mean"),
        평균_10위권일수=("days_in_top10", "mean"),
        평균_롱런지수=("롱런 지수", "mean")
    )
    .reindex(active_labels)
    .reset_index()
    .rename(columns={
        "영화 유형": "유형",
        "영화_편수": "영화 편수",
        "평균_스크린수": "평균 스크린 수 (개)",
        "평균_누적관객": "평균 누적 관객 (명)",
        "평균_10위권일수": "평균 10위권 일수",
        "평균_롱런지수": "평균 롱런 지수"
    })
)

summary["평균 스크린 수 (개)"] = (
    summary["평균 스크린 수 (개)"].round(1)
)
summary["평균 누적 관객 (명)"] = (
    summary["평균 누적 관객 (명)"].round(0)
)
summary["평균 10위권 일수"] = (
    summary["평균 10위권 일수"].round(1)
)
summary["평균 롱런 지수"] = (
    summary["평균 롱런 지수"].round(2)
)

st.dataframe(
    summary,
    hide_index=True,
    use_container_width=True
)

st.caption(
    "평균은 로그 변환 전 원래 단위로 계산합니다. "
    "유형 기호는 누적 관객 평균이 높은 순서로 부여됩니다."
)


# ==========================================
# 9. 유형별 누적 관객 상위 5편
# ==========================================
st.subheader("5. 유형별 누적 관객 상위 영화 5편")

for label in active_labels:
    st.markdown(f"### {label} 유형")

    top5 = (
        df[df["영화 유형"] == label]
        .sort_values("total_audi", ascending=False)
        .head(5)
        [["movieNm", "total_audi", "first_week_audi", "days_in_top10"]]
        .copy()
        .rename(columns={
            "movieNm": "영화 제목",
            "total_audi": "누적 관객 (명)",
            "first_week_audi": "첫 주 관객 (명)",
            "days_in_top10": "10위권 일수"
        })
    )

    if top5.empty:
        st.write("이 유형에 해당하는 영화가 없습니다.")
    else:
        top5["누적 관객 (명)"] = top5["누적 관객 (명)"].map(
            lambda value: f"{value:,.0f}"
        )
        top5["첫 주 관객 (명)"] = top5["첫 주 관객 (명)"].map(
            lambda value: f"{value:,.0f}"
        )

        st.dataframe(
            top5,
            hide_index=True,
            use_container_width=True
        )


# ==========================================
# 10. 묶음 수에 따른 SSE 계산
# ==========================================
st.subheader("6. 묶음 수 정하기: 군집 내 제곱거리 합")

st.write(
    "SSE는 각 영화와 해당 영화가 속한 군집 중심 사이의 "
    "거리 제곱을 모두 더한 값입니다. "
    "일반적으로 묶음 수가 늘어나면 SSE가 작아집니다."
)

sse_rows = []
sse_values = {}

max_k = min(7, len(df) - 1)

for k in range(1, max_k + 1):
    test_model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    test_model.fit(X_scaled)

    sse = float(test_model.inertia_)
    sse_values[k] = sse

    if k == 1:
        decrease = None
    else:
        decrease = sse_values[k - 1] - sse

    sse_rows.append({
        "묶음 수": k,
        "군집 내 제곱거리 합 (SSE)": sse,
        "직전 값 대비 감소량": decrease
    })

sse_df = pd.DataFrame(sse_rows)

fig_sse = go.Figure()

fig_sse.add_trace(
    go.Scatter(
        x=sse_df["묶음 수"],
        y=sse_df["군집 내 제곱거리 합 (SSE)"],
        mode="lines+markers",
        name="SSE",
        line={"width": 3},
        marker={"size": 8},
        hovertemplate=(
            "묶음 수: %{x}<br>"
            "SSE: %{y:,.2f}<extra></extra>"
        )
    )
)

# 현재 선택한 묶음 수 위치에 세로선
fig_sse.add_vline(
    x=chosen_k,
    line_dash="dash",
    line_width=2,
    line_color="red",
    annotation_text=f"현재 선택: {chosen_k}개",
    annotation_position="top"
)

fig_sse.update_layout(
    title="묶음 수에 따른 SSE 변화",
    xaxis_title="묶음 수",
    yaxis_title="군집 내 제곱거리 합 (SSE)",
    xaxis={
        "tickmode": "linear",
        "dtick": 1
    },
    height=450
)

st.plotly_chart(fig_sse, use_container_width=True)

# 표에 표시할 숫자 정리
sse_table = sse_df.copy()

sse_table["군집 내 제곱거리 합 (SSE)"] = (
    sse_table["군집 내 제곱거리 합 (SSE)"].map(
        lambda value: f"{value:,.2f}"
    )
)

sse_table["직전 값 대비 감소량"] = (
    sse_table["직전 값 대비 감소량"].map(
        lambda value: "" if pd.isna(value) else f"{value:,.2f}"
    )
)

st.dataframe(
    sse_table,
    hide_index=True,
    use_container_width=True
)

st.caption(
    "직전 값 대비 감소량은 바로 앞 묶음 수의 SSE에서 "
    "현재 묶음 수의 SSE를 뺀 값입니다. 첫 번째 행은 비교값이 없습니다."
)


# ==========================================
# 11. 실루엣 점수
# ==========================================
st.subheader("7. 현재 묶음의 뚜렷함: 실루엣 점수")

try:
    current_score = silhouette_score(
        X_scaled,
        df["군집 번호"]
    )

    st.markdown(
        f"**현재 묶음 수 {chosen_k}개 · "
        f"실루엣 점수: {current_score:.3f}**"
    )

    st.progress(
        max(0.0, min(1.0, (current_score + 1) / 2))
    )

    if current_score >= 0.7:
        st.write("해석: 군집이 비교적 뚜렷하게 분리되어 있습니다.")
    elif current_score >= 0.5:
        st.write("해석: 군집이 어느 정도 분리되어 있습니다.")
    elif current_score >= 0.25:
        st.write("해석: 군집 간 구분이 약하거나 일부 겹칠 수 있습니다.")
    else:
        st.write("해석: 군집이 많이 겹치거나 구분이 뚜렷하지 않을 수 있습니다.")

except Exception as error:
    st.warning(f"실루엣 점수를 계산할 수 없습니다: {error}")

st.caption(
    "실루엣 점수는 -1부터 1까지입니다. "
    "1에 가까울수록 같은 군집 안에서는 가깝고 다른 군집과는 "
    "멀리 떨어지는 경향이 있습니다. 0 근처는 군집이 겹칠 수 있고, "
    "음수는 일부 영화가 다른 군집에 더 가까울 수 있음을 뜻합니다."
)


# ==========================================
# 12. 분석 방법
# ==========================================
with st.expander("분석 방법 알아보기"):
    st.markdown("""
    **① 데이터 정리**

    분석에 필요한 값이 없거나 스크린 수, 누적 관객,
    첫 주 관객이 0 이하인 영화는 제외합니다.

    **② 네 가지 속성 계산**

    스크린 수와 누적 관객은 상용로그로 변환합니다.
    10위권 일수는 원래 값을 사용합니다.
    롱런 지수는 누적 관객을 첫 주 관객으로 나눈 값이며,
    20을 넘으면 20으로 제한합니다.

    **③ 표준화**

    선택한 속성들을 평균 0, 표준편차 1 기준으로 변환합니다.

    **④ K-평균**

    선택한 속성을 바탕으로 영화를 지정한 개수의 군집으로 나눕니다.
    정답 유형은 미리 주어지지 않습니다.

    **⑤ 유형 기호**

    누적 관객 평균이 높은 군집부터 ㉮, ㉯, ㉰, ㉱, ㉲, ㉳, ㉴ 순서로
    표시합니다. 기호는 영화의 공식 장르를 의미하지 않습니다.

    **⑥ SSE와 실루엣 점수**

    SSE는 군집 내부의 퍼짐을 나타냅니다.
    실루엣 점수는 같은 군집 안의 가까움과 다른 군집과의 분리 정도를
    함께 평가합니다. 두 지표와 산점도를 함께 살펴보는 것이 좋습니다.
    """)

st.caption(
    "같은 데이터와 속성 선택에 대해 random_state=42로 "
    "결과를 재현하도록 설정했습니다."
)
