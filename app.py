import os
import time
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from streamlit_plotly_events import plotly_events
import streamlit.components.v1 as components

st.set_page_config(layout="wide", page_title="IPL Auction Visualizer (Interactive)", page_icon="🏏")
st.title("IPL Auction Visualizer — Interactive & IPL Theme 🔥")

# -----------------------------------
# YOUR LOCAL MP3 FILE PATH (YOU ASKED FOR THIS)
# -----------------------------------
THEME_MP3_PATH = r"d:\Users\DELL\Downloads\ipl_theme.mp3"


# -----------------------------------
# CRICKET + IPL THEME ANIMATION
# -----------------------------------
def play_ipl_theme_with_animation(message="That’s a lovely shot!"):
    
    if os.path.exists(THEME_MP3_PATH):
        mp3_path = THEME_MP3_PATH.replace("\\", "/")

        html = f"""
        <style>
        .wrap {{ position:relative; width:100%; height:140px; overflow:hidden; }}
        .bat {{ font-size:52px; position:absolute; left:-30%; top:10px;
               animation:swing 0.8s ease-out 0s 1 forwards; }}
        .ball{{ font-size:48px; position:absolute; left:-10%; top:40px;
                animation:roll 0.8s ease-out 0s 1 forwards; }}
        @keyframes swing {{
            0%{{left:-30%; transform:rotate(-30deg); opacity:0}}
            30%{{left:8%; opacity:1}}
            100%{{left:110%; transform:rotate(20deg);}}
        }}
        @keyframes roll {{
            0%{{left:-10%; transform:rotate(0deg); opacity:0}}
            40%{{opacity:1}}
            100%{{left:120%; transform:rotate(720deg);}}
        }}
        .msg {{ font-weight:700; margin-top:6px; font-size:18px; text-align:center; }}
        .confetti {{ font-size:24px; text-align:center; margin-top:8px; }}
        </style>

        <div class="wrap">
            <div class="bat">🏏</div>
            <div class="ball">🏐</div>
        </div>

        <div class="msg">{message}</div>
        <div class="confetti">🎉🔥👏</div>

        <audio autoplay>
            <source src="file:///{mp3_path}" type="audio/mpeg">
        </audio>
        """

        components.html(html, height=260)

    else:
        components.html("<h4 style='color:red;'>IPL Theme MP3 not found at your path!</h4>", height=50)


# -----------------------------------
# SAMPLE DATASET (if no Excel uploaded)
# -----------------------------------
@st.cache_data
def generate_sample_dataset(n_rows=300, seed=42):
    rng = np.random.RandomState(seed)
    teams = ["CSK","MI","RCB","KKR","SRH","RR","DC","PBKS"]
    roles = ["Batsman","Bowler","All-Rounder","Wicket-Keeper"]
    players = [f"Player_{i}" for i in range(1,1501)]

    rows = []
    for i in range(n_rows):
        team = rng.choice(teams)
        role = rng.choice(roles, p=[0.35,0.33,0.22,0.10])
        purse = int(rng.randint(5,20) * 1_00_00_000)
        target = rng.choice(players)
        expected = int(rng.randint(50,200) * 1_00_000)
        prob = int(rng.randint(20,95))
        need = int(rng.randint(1,5))
        base = int(rng.choice([20,30,40,50,60,75,100]) * 1_00_000)

        rows.append({
            "Team":team,
            "Player_Role":role,
            "Purse_Remaining":purse,
            "Target_Player":target,
            "Expected_Bid":expected,
            "Buy_Probability(%)":prob,
            "Players_Needed":need,
            "Base_Price":base
        })

    return pd.DataFrame(rows)


# -----------------------------------
# DATASET UPLOADER
# -----------------------------------
st.sidebar.header("Upload IPL Dataset")
uploaded_file = st.sidebar.file_uploader("Upload Excel/CSV dataset", type=["xlsx","csv"])

if uploaded_file:
    try:
        if uploaded_file.name.endswith(".xlsx"):
            df = pd.read_excel(uploaded_file)
        else:
            df = pd.read_csv(uploaded_file)

        st.sidebar.success("Dataset loaded successfully!")
    except:
        st.sidebar.error("Failed to load file. Using sample dataset instead.")
        df = generate_sample_dataset()
else:
    df = generate_sample_dataset()
    st.sidebar.info("Using sample dataset.")

# Required columns check
required_cols = ["Team","Player_Role","Purse_Remaining",
                 "Target_Player","Expected_Bid",
                 "Buy_Probability(%)","Players_Needed","Base_Price"]

missing = [c for c in required_cols if c not in df.columns]

if missing:
    st.warning(f"Dataset missing columns: {missing}. Using sample dataset.")
    df = generate_sample_dataset()


# -----------------------------------
# ADD LAKH COLUMNS
# -----------------------------------
df["Expected_Bid_L"] = (df["Expected_Bid"] / 1_00_000).round(2)
df["Base_Price_L"] = (df["Base_Price"] / 1_00_000).round(2)

display_player_col = "Player_Name" if "Player_Name" in df.columns else "Target_Player"


# -----------------------------------
# FILTERS
# -----------------------------------
st.sidebar.header("Filters")

teams_list = sorted(df["Team"].unique())
teams_filter = st.sidebar.multiselect("Teams", teams_list, default=teams_list)

roles_list = sorted(df["Player_Role"].unique())
roles_filter = st.sidebar.multiselect("Roles", ["All"]+roles_list, default=["All"])


exp_min, exp_max = int(df["Expected_Bid_L"].min()), int(df["Expected_Bid_L"].max())
exp_range = st.sidebar.slider("Expected Bid Range (Lakhs ₹)", exp_min, exp_max, (exp_min, exp_max))

prob_threshold = st.sidebar.slider("Minimum Buy Probability (%)", 0, 100, 20)

main_metric = st.sidebar.selectbox("Main Chart Metric", [
    "Total Players Needed",
    "Unique Targets",
    "Avg Expected Bid (L)"
])


# -----------------------------------
# APPLY FILTERS
# -----------------------------------
df_filtered = df.copy()

if teams_filter:
    df_filtered = df_filtered[df_filtered["Team"].isin(teams_filter)]

if "All" not in roles_filter:
    df_filtered = df_filtered[df_filtered["Player_Role"].isin(roles_filter)]

df_filtered = df_filtered[
    (df_filtered["Expected_Bid_L"] >= exp_range[0]) &
    (df_filtered["Expected_Bid_L"] <= exp_range[1])
]

df_filtered = df_filtered[df_filtered["Buy_Probability(%)"] >= prob_threshold]


# -----------------------------------
# AGGREGATION
# -----------------------------------
if not df_filtered.empty:
    agg = df_filtered.groupby("Team").agg(
        Total_Players_Needed=("Players_Needed","sum"),
        Unique_Targets=(display_player_col, lambda x: x.nunique()),
        Avg_Expected_Bid_L=("Expected_Bid_L","mean")
    ).reset_index()
else:
    agg = pd.DataFrame()


# -----------------------------------
# MAIN CHART
# -----------------------------------
st.subheader("IPL Team Demand Overview")

if agg.empty:
    st.info("No data available with current filters.")
else:
    if main_metric == "Total Players Needed":
        fig = px.bar(agg, x="Team", y="Total_Players_Needed", text="Total_Players_Needed", color="Team")
    elif main_metric == "Unique Targets":
        fig = px.bar(agg, x="Team", y="Unique_Targets", text="Unique_Targets", color="Team")
    else:
        fig = px.bar(agg, x="Team", y="Avg_Expected_Bid_L", text="Avg_Expected_Bid_L", color="Team",
                     labels={"Avg_Expected_Bid_L": "Avg Expected Bid (Lakhs ₹)"})

    fig.update_traces(textposition="outside")
    st.plotly_chart(fig, use_container_width=True)


# -----------------------------------
# ANIMATION BUTTON
# -----------------------------------
if st.button("Play IPL Animation 🎵🔥"):
    play_ipl_theme_with_animation("IPL Theme Playing — Enjoy the Auction Vibes! 🎉🔥")


# -----------------------------------
# SCATTER PLOT
# -----------------------------------
st.subheader("Expected Bid vs Buy Probability")

if not df_filtered.empty:
    scatter = px.scatter(
        df_filtered,
        x="Expected_Bid_L",
        y="Buy_Probability(%)",
        color="Team",
        hover_data=[display_player_col, "Player_Role", "Base_Price_L"],
        labels={"Expected_Bid_L":"Expected Bid (Lakhs ₹)"}
    )
    st.plotly_chart(scatter, use_container_width=True)
else:
    st.info("No scatter data available.")


# -----------------------------------
# DOWNLOAD
# -----------------------------------
@st.cache_data
def to_csv(df):
    return df.to_csv(index=False).encode("utf-8")

csv = to_csv(df_filtered)
st.download_button("Download Filtered Data as CSV", csv, "ipl_filtered.csv", "text/csv")
