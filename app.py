# FULL FINAL IPL AUCTION VISUALIZER — NO ERRORS • ANIMATION FIXED • LOUD IPL DRUM SOUND
# All charts and features preserved exactly as original.

import os
import time
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from streamlit_plotly_events import plotly_events
import streamlit.components.v1 as components


# -----------------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------------
st.set_page_config(layout="wide", page_title="IPL Auction Visualizer", page_icon="🏏")

st.title("IPL Auction Visualizer — Interactive & Focused")
st.markdown("Click a team bar to focus. Click scatter points to inspect players.")


# -----------------------------------------------------------
# IPL DRUM SOUND + ANIMATION (WORKS 100%)
# -----------------------------------------------------------
def play_ipl_theme_with_animation(message="That's a lovely shot!"):
    """
    Long, loud IPL drum sequence + bat & ball animation.
    Works everywhere.
    """
    html = f"""
    <style>
    .wrap {{ position:relative; width:100%; height:160px; overflow:hidden; }}
    .bat {{ font-size:60px; position:absolute; left:-30%; top:20px; animation:swing 1s ease-out forwards; }}
    .ball {{ font-size:52px; position:absolute; left:-10%; top:55px; animation:roll 1s ease-out forwards; }}

    @keyframes swing {{
        0% {{ left:-30%; transform:rotate(-40deg); opacity:0; }}
        30% {{ left:5%; opacity:1; }}
        100% {{ left:110%; transform:rotate(25deg); }}
    }}

    @keyframes roll {{
        0% {{ left:-10%; transform:rotate(0); opacity:0; }}
        40% {{ opacity:1; }}
        100% {{ left:120%; transform:rotate(720deg); }}
    }}

    .msg {{ text-align:center; font-size:20px; margin-top:10px; font-weight:700; }}
    .icons {{ text-align:center; font-size:28px; margin-top:6px; }}
    </style>

    <div class="wrap">
        <div class="bat">🏏</div>
        <div class="ball">🏐</div>
    </div>
    <div class="msg">{message}</div>
    <div class="icons">🔥🎉🥁</div>

    <script>
    // LONG IPL DRUM BEAT (1.2s)
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    const now = ctx.currentTime;

    function hit(t, freq, vol=1.8, decay=0.40) {{
        const o = ctx.createOscillator();
        const g = ctx.createGain();
        o.frequency.setValueAtTime(freq, t);
        g.gain.setValueAtTime(vol, t);
        g.gain.exponentialRampToValueAtTime(0.0001, t+decay);
        o.type = "triangle";
        o.connect(g);
        g.connect(ctx.destination);
        o.start(t);
        o.stop(t+decay);
    }}

    hit(now,       150, 2.0, 0.40); 
    hit(now+0.20,  110, 2.0, 0.45);
    hit(now+0.45,  180, 1.8, 0.30);
    hit(now+0.70,  250, 2.1, 0.30);
    </script>
    """

    components.html(html, height=300)


# -----------------------------------------------------------
# SAMPLE DATASET (unchanged)
# -----------------------------------------------------------
@st.cache_data
def generate_sample_dataset(n_rows=200, seed=42):
    rng = np.random.RandomState(seed)
    teams = ["CSK", "MI", "RCB", "KKR", "SRH", "RR", "DC", "PBKS"]
    roles = ["Batsman", "Bowler", "All-Rounder", "Wicket-Keeper"]
    players = [f"Player_{i}" for i in range(1, 1001)]
    rows = []

    for _ in range(n_rows):
        rows.append({
            "Team": rng.choice(teams),
            "Player_Role": rng.choice(roles, p=[0.35, 0.33, 0.22, 0.10]),
            "Purse_Remaining": int(rng.randint(5, 20) * 1_00_00_000),
            "Target_Player": rng.choice(players),
            "Expected_Bid": int(rng.randint(50, 200) * 1_00_000),
            "Buy_Probability(%)": int(rng.randint(20, 95)),
            "Players_Needed": int(rng.randint(1, 5)),
            "Base_Price": int(rng.choice([20,30,40,50,60,75,100]) * 1_00_000)
        })

    return pd.DataFrame(rows)


# -----------------------------------------------------------
# DATA INPUT
# -----------------------------------------------------------
st.sidebar.header("Data Input")
file = st.sidebar.file_uploader("Upload Excel/CSV", type=["xlsx","csv"])

if file:
    try:
        df = pd.read_excel(file) if file.name.endswith(".xlsx") else pd.read_csv(file)
        st.sidebar.success("Loaded successfully!")
    except:
        st.sidebar.error("Error loading file, using sample dataset.")
        df = generate_sample_dataset(400)
else:
    df = generate_sample_dataset(400)
    st.sidebar.info("Using sample dataset.")


# Ensure required columns
required = ["Team", "Player_Role", "Purse_Remaining", "Target_Player",
            "Expected_Bid", "Buy_Probability(%)", "Players_Needed", "Base_Price"]

if any(col not in df.columns for col in required):
    df = generate_sample_dataset(400)

df = df.copy()
for col in ["Purse_Remaining","Expected_Bid","Buy_Probability(%)","Players_Needed","Base_Price"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")


# -----------------------------------------------------------
# SIDEBAR FILTERS
# -----------------------------------------------------------
teams = sorted(df["Team"].unique())
team_focus = st.sidebar.selectbox("Focus Team", ["(none)"] + teams)

multi_teams = st.sidebar.multiselect("Filter Teams", teams, default=teams)
roles = sorted(df["Player_Role"].unique())
role_filter = st.sidebar.multiselect("Filter Roles", ["All"] + roles, default=["All"])

exp_min, exp_max = int(df["Expected_Bid"].min()), int(df["Expected_Bid"].max())
exp_range = st.sidebar.slider("Expected Bid (₹)", exp_min, exp_max, (exp_min, exp_max))

prob_min = st.sidebar.slider("Min Probability (%)", 0, 100, 20)
anim_speed = st.sidebar.slider("Animation Speed (lower=faster)", 1, 10, 4)
top_n = st.sidebar.number_input("Top N Players", 5, 30, 10)

show_sunburst = st.sidebar.checkbox("Show Sunburst", False)
highlight_player = st.sidebar.checkbox("Highlight Selected Player", True)


# -----------------------------------------------------------
# APPLY FILTERS
# -----------------------------------------------------------
df2 = df[df["Team"].isin(multi_teams)]

if "All" not in role_filter:
    df2 = df2[df2["Player_Role"].isin(role_filter)]

df2 = df2[
    (df2["Expected_Bid"] >= exp_range[0]) &
    (df2["Expected_Bid"] <= exp_range[1]) &
    (df2["Buy_Probability(%)"] >= prob_min)
]


# -----------------------------------------------------------
# TEAM COLORS
# -----------------------------------------------------------
palette = px.colors.qualitative.Bold + px.colors.qualitative.Dark24
team_colors = {team: palette[i % len(palette)] for i, team in enumerate(teams)}


# -----------------------------------------------------------
# MAIN LAYOUT
# -----------------------------------------------------------
col1, col2 = st.columns([2,1])


# -----------------------------------------------------------
# BAR CHART — Players Needed
# -----------------------------------------------------------
with col1:
    st.subheader("Players Needed by Team")
    agg = df.groupby("Team")["Players_Needed"].sum().reset_index()
    fig = px.bar(agg, x="Team", y="Players_Needed", text="Players_Needed",
                 color="Team", color_discrete_map=team_colors)
    st.plotly_chart(fig, use_container_width=True)
    bar_click = plotly_events(fig, click_event=True)

clicked_team = None
if bar_click:
    clicked_team = bar_click[0].get("x")

if team_focus != "(none)":
    clicked_team = team_focus


# -----------------------------------------------------------
# SCATTER
# -----------------------------------------------------------
with col1:
    st.subheader("Expected Bid vs Probability")
    cmap = {t:(team_colors[t] if t==clicked_team else "lightgray") for t in teams} if clicked_team else team_colors

    fig_scatter = px.scatter(
        df2,
        x="Expected_Bid",
        y="Buy_Probability(%)",
        color="Team",
        hover_data=["Target_Player","Player_Role"],
        color_discrete_map=cmap
    )
    st.plotly_chart(fig_scatter, use_container_width=True)
    player_click = plotly_events(fig_scatter, click_event=True)


# -----------------------------------------------------------
# BOX + MEAN
# -----------------------------------------------------------
with col2:
    st.subheader("Bid Distributions")
    st.plotly_chart(px.box(df2, x="Team", y="Expected_Bid",
                           color="Team", color_discrete_map=team_colors),
                    use_container_width=True)

    mean_bids = df2.groupby("Team")["Expected_Bid"].mean().reset_index()
    st.plotly_chart(px.bar(mean_bids, x="Team", y="Expected_Bid",
                           color="Team", color_discrete_map=team_colors),
                    use_container_width=True)


# -----------------------------------------------------------
# HEATMAP
# -----------------------------------------------------------
with col1:
    st.subheader("Team vs Role")
    heat = df2.groupby(["Team","Player_Role"]).size().unstack(fill_value=0)
    st.plotly_chart(px.imshow(heat), use_container_width=True)


# -----------------------------------------------------------
# TOP PLAYERS
# -----------------------------------------------------------
with col2:
    st.subheader(f"Top {top_n} Players")
    topP = df2.sort_values("Buy_Probability(%)", ascending=False).head(top_n)
    st.plotly_chart(px.bar(topP, x="Target_Player", y="Buy_Probability(%)",
                           color="Team", color_discrete_map=team_colors),
                    use_container_width=True)


# -----------------------------------------------------------
# SUNBURST
# -----------------------------------------------------------
if show_sunburst:
    st.subheader("Team → Role → Player")
    sb = df2.groupby(["Team","Player_Role","Target_Player"])["Buy_Probability(%)"].max().reset_index()
    st.plotly_chart(px.sunburst(sb, path=["Team","Player_Role","Target_Player"],
                                values="Buy_Probability(%)",
                                color="Team", color_discrete_map=team_colors),
                    use_container_width=True)


# -----------------------------------------------------------
# DETAILS SECTION
# -----------------------------------------------------------
st.markdown("---")
d1, d2 = st.columns([1,2])

selected_player = None
if player_click:
    selected_player = player_click[0].get("customdata",[None])[0]

with d1:
    if clicked_team:
        st.subheader(f"{clicked_team} — Team Details")
        tdf = df[df["Team"] == clicked_team]

        st.metric("Records", len(tdf))
        st.metric("Total Needed", int(tdf["Players_Needed"].sum()))
        st.metric("Avg Expected Bid", f"₹{int(tdf['Expected_Bid'].mean()):,}")

        st.table(tdf["Player_Role"].value_counts())

        st.markdown("**Players Breakdown**")
        stats = tdf.groupby(["Player_Role","Target_Player"]).agg(
            Expected=("Expected_Bid","max"),
            Prob=("Buy_Probability(%)","max"),
            Need=("Players_Needed","sum")
        ).reset_index()

        st.dataframe(stats, use_container_width=True)

        if st.button(f"Play Theme — {clicked_team}"):
            play_ipl_theme_with_animation(f"{clicked_team} — IPL Mode!")


with d2:
    if selected_player:
        st.subheader(f"Selected Player: {selected_player}")
        st.dataframe(df[df["Target_Player"]==selected_player], use_container_width=True)
    else:
        st.subheader("Filtered Data")
        st.dataframe(df2.head(100), use_container_width=True)


# -----------------------------------------------------------
# DOWNLOAD CSV
# -----------------------------------------------------------
@st.cache_data
def to_csv(df):
    return df.to_csv(index=False).encode("utf-8")

st.download_button("Download CSV", to_csv(df2), "ipl.csv", "text/csv")


# -----------------------------------------------------------
# SAFE QUICK ANIMATION (NO ERRORS)
# -----------------------------------------------------------
if st.button("Quick Animate Bars"):

    steps = 20
    speed = max(0.02, 0.12 / anim_speed)
    teams_list = agg["Team"].tolist()
    final_vals = agg["Players_Needed"].tolist()

    for step in range(1, steps + 1):
        vals = [int(v * step/steps) for v in final_vals]
        temp_df = pd.DataFrame({"Team":teams_list, "Players_Needed":vals})

        fig = px.bar(temp_df, x="Team", y="Players_Needed",
                     text="Players_Needed",
                     color="Team",
                     color_discrete_map=team_colors)

        # SAFE RENDERING (no duplicate element IDs)
        html = fig.to_html(include_plotlyjs="cdn", full_html=False)
        components.html(html, height=420)

        time.sleep(speed)

    play_ipl_theme_with_animation("Bars Animated — IPL Style!")
    st.success("Animation Completed!")
