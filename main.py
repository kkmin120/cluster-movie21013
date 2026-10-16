
import math

import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


# ==========================================
# 1. 화면 설정
# ==========================================
st.set_page_config(
    page_title="영화 유형 나누기",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 영화 유형 나누기")
st.write(
    "영화의 특징을 분석하여 비슷한 영화끼리 "
    "세 가지 유형으로 나누는 프로그램입니다."
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

COLORS = {
    "㉮": "#E76F51",
    "㉯": "#457B9D",
    "㉰": "#2A9D8F"
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
            "CSV 파일에 필요한 열이 없습니다: " + ", ".join(missing)
        )

    return data


try:
    original = load_data()
except Exception as error:
    st.error("영화 데이터를 불러오지 못했습니다.")
    st.code(str(error))
    st.info("데이터 주소와 인터넷 연결을 확인해 주세요.")
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

# 필수 숫자 속성이 모두 있고 양수 조건을 만족하는 영화만 사용
valid = (
    df[number_columns].notna().all(axis=1)
    & (df["first_scrn"] > 0)
    & (df["total_audi"] > 0)
    & (df["first_week_audi"] > 0)
)

df = df.loc[valid].copy()

# 로그 속성은 상용로그
df["스크린 수 (로그)"] = df["first_scrn"].map(math.log10)
df["누적 관객 (로그)"] = df["total_audi"].map(math.log10)

# 10위권 일수는 원래 값 사용
df["10위권 일수"] = df["days_in_top10"]

# 롱런 지수: 누적 관객 / 첫 주 관객, 최대 20
df["롱런 지수"] = (
    df["total_audi"] / df["first_week_audi"]
).clip(upper=20)

df = df.reset_index(drop=True)

st.markdown(
    f"**전체 영화: {total_count:,}편　|　"
    f"분석에 사용한 영화: {len(df):,}편**"
)

st.caption(
    "필수 값이 없거나 스크린 수, 누적 관객, 첫 주 관객이 "
    "0 이하인 영화는 분석에서 제외합니다."
)

if len(df) < 3:
    st.error("분석에 사용할 영화가 3편보다 적어 유형을 나눌 수 없습니다.")
    st.stop()


# ==========================================
# 4. 군집 분석에 사용할 속성 선택
# ==========================================
st.subheader("1. 유형 분류에 사용할 속성")

selected = st.multiselect(
    "두 개 이상 선택하세요.",
    options=FEATURES,
    default=FEATURES,
    key="features"
)

if len(selected) < 2:
    st.warning("군집 분석을 위해 속성을 두 개 이상 선택해 주세요.")
    st.stop()


# ==========================================
# 5. 표준화 및 K-평균
# ==========================================
X = df[selected].astype(float)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

df["군집"] = model.fit_predict(X_scaled)

# 누적 관객 평균이 높은 군집부터 ㉮, ㉯, ㉰ 부여
cluster_means = (
    df.groupby("군집")["total_audi"]
    .mean()
    .sort_values(ascending=False)
)

labels = ["㉮", "㉯", "㉰"]
label_map = {
    cluster_id: labels[i]
    for i, cluster_id in enumerate(cluster_means.index)
}

df["영화 유형"] = df["군집"].map(label_map)


# ==========================================
# 6. 2차원 산점도
# ==========================================
st.subheader("2. 2차원 산점도")
st.write("가로축과 세로축을 바꾸어 영화 유형의 분포를 살펴보세요.")

left, right = st.columns(2)

with left:
    x_axis = st.selectbox(
        "가로축 속성",
        selected,
        index=0,
        key="x_axis"
    )

y_choices = [item for item in selected if item != x_axis]

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
    category_orders={"영화 유형": labels},
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
    title="영화 유형별 2차원 분포"
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
        "3차원 산점도를 보려면 위에서 속성을 세 개 이상 선택해 주세요."
    )
else:
    st.write(
        "X·Y·Z축을 선택하세요. 그래프를 마우스로 드래그하면 "
        "회전할 수 있으며, 점에 마우스를 올리면 영화 제목이 표시됩니다."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        x3 = st.selectbox(
            "X축",
            selected,
            index=0,
            key="x3"
        )

    y3_choices = [item for item in selected if item != x3]

    with col2:
        y3 = st.selectbox(
            "Y축",
            y3_choices,
            index=0,
            key="y3"
        )

    z3_choices = [
        item for item in selected
        if item != x3 and item != y3
    ]

    with col3:
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
        category_orders={"영화 유형": labels},
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
        title="영화 유형별 3차원 분포"
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
    .agg({
        "movieNm": "size",
        "first_scrn": "mean",
        "total_audi": "mean",
        "days_in_top10": "mean",
        "롱런 지수": "mean"
    })
    .rename(columns={
        "movieNm": "영화 편수",
        "first_scrn": "평균 스크린 수 (개)",
        "total_audi": "평균 누적 관객 (명)",
        "days_in_top10": "평균 10위권 일수",
        "롱런 지수": "평균 롱런 지수"
    })
    .reindex(labels)
    .reset_index()
    .rename(columns={"영화 유형": "유형"})
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
    "평균은 로그 변환 전의 원래 단위로 계산합니다. "
    "㉮는 누적 관객 평균이 가장 높은 유형입니다."
)


# ==========================================
# 9. 유형별 누적 관객 상위 5편
# ==========================================
st.subheader("5. 유형별 누적 관객 상위 영화 5편")

for label in labels:
    st.markdown(f"### {label} 유형")

    top5 = (
        df[df["영화 유형"] == label]
        .sort_values("total_audi", ascending=False)
        .head(5)
        .loc[:, [
            "movieNm",
            "total_audi",
            "first_week_audi",
            "days_in_top10"
        ]]
        .copy()
    )

    top5 = top5.rename(columns={
        "movieNm": "영화 제목",
        "total_audi": "누적 관객 (명)",
        "first_week_audi": "첫 주 관객 (명)",
        "days_in_top10": "10위권 일수"
    })

    if top5.empty:
        st.write("해당 유형에 영화가 없습니다.")
    else:
        top5["누적 관객 (명)"] = (
            top5["누적 관객 (명)"].map(lambda n: f"{n:,.0f}")
        )
        top5["첫 주 관객 (명)"] = (
            top5["첫 주 관객 (명)"].map(lambda n: f"{n:,.0f}")
        )

        st.dataframe(
            top5,
            hide_index=True,
            use_container_width=True
        )


# ==========================================
# 10. 분석 방법
# ==========================================
with st.expander("분석 방법 알아보기"):
    st.markdown("""
    **① 데이터 정리**

    분석에 필요한 값이 없는 영화와 스크린 수, 누적 관객,
    첫 주 관객이 0 이하인 영화는 제외합니다.

    **② 속성 계산**

    스크린 수와 누적 관객에는 상용로그를 적용합니다.
    10위권 일수는 그대로 사용합니다.
    롱런 지수는 누적 관객을 첫 주 관객으로 나눈 값이며,
    최댓값은 20입니다.

    **③ 표준화**

    서로 다른 단위의 속도를 비교할 수 있도록 각 속성을
    평균 0, 표준편차 1 기준으로 변환합니다.

    **④ K-평균**

    선택한 속성의 유사성을 바탕으로 영화 세 유형을 찾습니다.
    미리 정해진 정답 유형은 사용하지 않습니다.

    **⑤ 유형 이름**

    누적 관객 평균이 높은 군집부터 ㉮, ㉯, ㉰로 표시합니다.
    이 기호는 영화의 장르나 공식 분류를 뜻하지 않습니다.
    """)

st.caption(
    "같은 데이터와 같은 속성 선택에 대해 random_state=42로 "
    "군집 결과를 재현하도록 설정했습니다."
)
