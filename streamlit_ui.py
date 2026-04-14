import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


DATA_PATH = "cleaned_football_dataset.csv"
POSITION_MAP = {
    "FW": 3,
    "MF": 2,
    "DF": 1,
    "GK": 0,
}
MATCH_FEATURES = ["Shots", "xG", "Matches"]
PLAYER_FEATURES = ["Age", "Position_Code", "Matches", "Minutes", "Assists", "xAG", "Shots"]


def set_page_style() -> None:
    st.set_page_config(page_title="Football Prediction Studio", layout="wide")
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;600;700;800&display=swap');

        :root {
            --ink: #0f172a;
            --accent: #ff6b4a;
            --accent-dark: #d94f31;
            --mint: #2cb67d;
            --paper: #f7fbfc;
            --card: rgba(255, 255, 255, 0.78);
            --line: rgba(15, 23, 42, 0.10);
        }

        .stApp {
            font-family: 'Sora', sans-serif;
            background:
                radial-gradient(circle at 8% 12%, rgba(44, 182, 125, 0.14), transparent 28%),
                radial-gradient(circle at 88% 8%, rgba(255, 107, 74, 0.14), transparent 24%),
                linear-gradient(180deg, #eef6fb 0%, #f8fbff 48%, #f4f8fc 100%);
            color: var(--ink);
        }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 2.5rem;
            max-width: 1240px;
        }

        .stApp p,
        .stApp span,
        .stApp label,
        .stApp h2,
        .stApp h3 {
            color: #000000 !important;
        }

        header[data-testid="stHeader"] * {
            color: #ffffff !important;
        }

        div[data-testid="stToolbar"] * {
            color: #ffffff !important;
        }

        .hero {
            border-radius: 24px;
            padding: 1.6rem 1.6rem;
            margin-bottom: 1.2rem;
            background:
                radial-gradient(circle at top right, rgba(255,255,255,0.12), transparent 22%),
                linear-gradient(135deg, #0f4c81 0%, #126e82 58%, #159895 100%);
            box-shadow: 0 18px 40px rgba(16, 42, 67, 0.18);
            color: #f4f9ff;
            border: 1px solid rgba(255, 255, 255, 0.18);
        }

        .hero h1,
        .hero p {
            color: #f4f9ff !important;
        }

        .hero h1 {
            margin: 0;
            font-size: 1.85rem;
            letter-spacing: 0.2px;
        }

        .hero p {
            margin-top: 0.35rem;
            margin-bottom: 0;
            opacity: 0.95;
        }

        .section-card {
            background: var(--card);
            border: 1px solid var(--line);
            border-radius: 18px;
            padding: 1rem 1.1rem;
            box-shadow: 0 14px 32px rgba(15, 23, 42, 0.06);
            backdrop-filter: blur(8px);
            margin-bottom: 1rem;
        }

        .section-title {
            font-size: 1.25rem;
            font-weight: 700;
            margin-bottom: 0.2rem;
        }

        .section-copy {
            color: #475569 !important;
            font-size: 0.95rem;
            margin-bottom: 0;
        }

        div[data-testid="metric-container"] {
            border-radius: 18px;
            border: 1px solid var(--line);
            background: var(--card);
            padding: 0.4rem 0.5rem;
            box-shadow: 0 10px 24px rgba(15, 23, 42, 0.05);
        }

        div[data-testid="metric-container"] label {
            font-size: 0.82rem;
            letter-spacing: 0.2px;
        }

        div[data-testid="metric-container"] [data-testid="stMetricValue"] {
            font-size: 1.85rem;
            font-weight: 700;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 0.45rem;
            margin-bottom: 1rem;
        }

        .stTabs [data-baseweb="tab"] {
            border-radius: 999px;
            padding: 0.45rem 0.95rem;
            background: rgba(255, 255, 255, 0.65);
            border: 1px solid var(--line);
        }

        .stTabs [aria-selected="true"] {
            background: linear-gradient(90deg, rgba(255,107,74,0.12), rgba(21,152,149,0.12));
            border-color: rgba(255, 107, 74, 0.24);
        }

        .stNumberInput > div,
        .stSelectbox > div[data-baseweb="select"] {
            border-radius: 14px;
        }

        .stNumberInput input {
            background: rgba(255, 255, 255, 0.86);
            color: #000000 !important;
            -webkit-text-fill-color: #000000 !important;
            caret-color: #000000 !important;
        }

        .stNumberInput button,
        .stNumberInput button span {
            color: #ffffff !important;
        }

        div[data-testid="stForm"] {
            border: 1px solid var(--line);
            border-radius: 18px;
            padding: 1rem;
            background: var(--card);
        }

        .stButton > button {
            border-radius: 999px;
            border: none;
            padding: 0.55rem 1.2rem;
            font-weight: 700;
            background: linear-gradient(90deg, var(--accent) 0%, #ff8a5b 100%);
            color: #ffffff;
            box-shadow: 0 10px 22px rgba(255, 107, 74, 0.24);
            transition: transform 0.18s ease, box-shadow 0.18s ease;
        }

        .stButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 12px 26px rgba(255, 107, 74, 0.30);
        }

        @media (max-width: 900px) {
            .hero h1 { font-size: 1.45rem; }
            .block-container { padding-top: 1.2rem; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data
def load_dataset(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def prepare_player_df(df: pd.DataFrame) -> pd.DataFrame:
    player_df = df.copy()
    for col in ["Age", "Matches", "Minutes", "Goals", "Assists", "xAG", "Shots"]:
        player_df[col] = player_df[col].fillna(player_df[col].median())
    if player_df["Position"].isnull().sum() > 0:
        player_df["Position"] = player_df["Position"].fillna(player_df["Position"].mode()[0])
    player_df["Position_Code"] = player_df["Position"].map(POSITION_MAP).fillna(0)
    return player_df


@st.cache_resource
def train_match_model(df: pd.DataFrame):
    X = df[MATCH_FEATURES]
    y = df["Goals"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    train_r2 = r2_score(y_train, model.predict(X_train))
    test_r2 = r2_score(y_test, model.predict(X_test))
    return model, train_r2, test_r2


@st.cache_resource
def train_player_model(player_df: pd.DataFrame):
    X = player_df[PLAYER_FEATURES]
    y = player_df["Goals"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    train_r2 = r2_score(y_train, model.predict(X_train))
    test_r2 = r2_score(y_test, model.predict(X_test))
    return model, train_r2, test_r2


def render_dashboard(df: pd.DataFrame) -> None:
    st.markdown(
        """
        <div class="section-card">
            <div class="section-title">Data Snapshot</div>
            <p class="section-copy">A quick view of scoring volume, shot activity, and relationships inside the dataset.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", len(df))
    c2.metric("Avg Goals", f"{df['Goals'].mean():.2f}")
    c3.metric("Avg Shots", f"{df['Shots'].mean():.2f}")
    c4.metric("Avg xG", f"{df['xG'].mean():.2f}")

    left, right = st.columns(2)
    with left:
        st.subheader("Shots vs Goals")
        fig_scatter, ax_scatter = plt.subplots(figsize=(6, 4.5))
        sns.scatterplot(data=df, x="Shots", y="Goals", color="#f25f4c", ax=ax_scatter, s=52)
        ax_scatter.grid(alpha=0.2)
        st.pyplot(fig_scatter)

    with right:
        st.subheader("Correlation Matrix")
        numeric_cols = ["Goals", "Assists", "xG", "xAG", "Shots", "Shots_on_Target"]
        fig_heat, ax_heat = plt.subplots(figsize=(6, 4.5))
        sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="RdYlBu_r", ax=ax_heat)
        st.pyplot(fig_heat)


def render_match_prediction(match_model: LinearRegression) -> None:
    st.markdown(
        """
        <div class="section-card">
            <div class="section-title">Match Prediction</div>
            <p class="section-copy">Estimate expected goals for each side using shots, xG, and matches played, then compare the result.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    home_col, away_col = st.columns(2)
    with home_col:
        st.markdown("### Home Team")
        home_shots = st.number_input("Home shots", min_value=0, value=11, step=1, key="h_shots")
        home_xg = st.number_input("Home xG", min_value=0.0, value=1.6, step=0.1, format="%g", key="h_xg")
        home_matches = st.number_input("Home matches", min_value=1, value=15, step=1, key="h_match")

    with away_col:
        st.markdown("### Away Team")
        away_shots = st.number_input("Away shots", min_value=0, value=9, step=1, key="a_shots")
        away_xg = st.number_input("Away xG", min_value=0.0, value=1.2, step=0.1, format="%g", key="a_xg")
        away_matches = st.number_input("Away matches", min_value=1, value=15, step=1, key="a_match")

    if st.button("Predict Match Outcome", use_container_width=True):
        home_data = pd.DataFrame([[home_shots, home_xg, home_matches]], columns=MATCH_FEATURES)
        away_data = pd.DataFrame([[away_shots, away_xg, away_matches]], columns=MATCH_FEATURES)

        pred_home = max(0.0, float(match_model.predict(home_data)[0]))
        pred_away = max(0.0, float(match_model.predict(away_data)[0]))

        if pred_home > pred_away:
            outcome = "Home Win"
        elif pred_away > pred_home:
            outcome = "Away Win"
        else:
            outcome = "Draw"

        r1, r2, r3 = st.columns(3)
        r1.metric("Pred Home Goals", f"{pred_home:.2f}")
        r2.metric("Pred Away Goals", f"{pred_away:.2f}")
        r3.metric("Outcome", outcome)


def render_player_prediction(player_model: LinearRegression) -> None:
    st.markdown(
        """
        <div class="section-card">
            <div class="section-title">Player Goal Prediction</div>
            <p class="section-copy">Project a player's likely goal output from profile, playing time, creation, and shooting data.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Player age", min_value=16, max_value=45, value=24, step=1)
        position = st.selectbox("Position", ["FW", "MF", "DF", "GK"])
        matches = st.number_input("Matches played", min_value=1, max_value=70, value=25, step=1)
        minutes = st.number_input("Minutes played", min_value=0, max_value=6000, value=1800, step=1)

    with col2:
        assists = st.number_input("Assists", min_value=0, max_value=50, value=6, step=1)
        xag = st.number_input("xAG", min_value=0.0, max_value=50.0, value=4.2, step=0.1, format="%g")
        shots = st.number_input("Shots", min_value=0, max_value=300, value=48, step=1)

    if st.button("Predict Player Goals", use_container_width=True):
        player_input = pd.DataFrame(
            [[age, POSITION_MAP[position], matches, minutes, assists, xag, shots]],
            columns=PLAYER_FEATURES,
        )
        pred_goals = max(0.0, float(player_model.predict(player_input)[0]))
        st.success(f"Predicted player goals: {pred_goals:.2f}")


def main() -> None:
    set_page_style()

    df = load_dataset(DATA_PATH)
    player_df = prepare_player_df(df)

    match_model, match_train_r2, match_test_r2 = train_match_model(df)
    player_model, player_train_r2, player_test_r2 = train_player_model(player_df)

    st.markdown(
        """
        <div class="hero">
            <h1>Football Prediction Studio</h1>
            <p>Notebook-based Streamlit app for match outcome and player goal prediction with a cleaner dashboard feel.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Match Train R2", f"{match_train_r2:.3f}")
    m2.metric("Match Test R2", f"{match_test_r2:.3f}")
    m3.metric("Player Train R2", f"{player_train_r2:.3f}")
    m4.metric("Player Test R2", f"{player_test_r2:.3f}")

    tab_overview, tab_match, tab_player = st.tabs(["Overview", "Match Predictor", "Player Predictor"])

    with tab_overview:
        render_dashboard(df)

    with tab_match:
        render_match_prediction(match_model)

    with tab_player:
        render_player_prediction(player_model)


if __name__ == "__main__":
    main()
