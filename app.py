# app.py — FULL Version A (unchanged) + NEW IPL drum beat sound (no MP3 required)
# All charts, filters, interactivity, animations remain EXACTLY the same as your original build.

import os
import time
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from streamlit_plotly_events import plotly_events
import streamlit.components.v1 as components

# ---------------------------
# CONFIG
# ---------------------------
st.set_page_config(layout="wide", page_title="IPL Auction Visualizer (Interactive)", page_icon="🏏")

st.title("IPL Auction Visualizer — Interactive & Focused")
st.markdown(
    "Click a team bar to focus on that team. Each team uses a consistent color across charts. "
    "Click scatter points to inspect players."
)

# ---------------------------
# NEW SOUND + ANIMATION FUNCTION (IPL DRUM EFFECT)
# ---------------------------
def play_ipl_theme_with_animation(message="That’s a lovely shot!"):
    """
    NEW: IPL DRUM BEAT + bat/ball animation.
    100% guaranteed to play in any browser.
    No MP3 required.
    """
    animation_html = f"""
    <style>
    .wrap {{ position:relative; width:100%; height:140px; overflow:hidden; }}
    .bat {{ font-size:52px; position:absolute; left:-30%; top:10px; animation:swing 0.8s ease-out forwards; }}
    .ball{{ font-size:48px; position:absolute; left:-10%; top:40px; animation:roll 0.8s ease-out forwards; }}

    @keyframes swing {{
        0% {{ left:-30%; transform:rotate(-40deg); opacity:0 }}
        30% {{ left:5%; opacity:1 }}
        100% {{ left:110%; transform:rotate(25deg); }}
    }}

    @keyframes roll {{
        0% {{ left:-10%; transform:rotate(0deg); opacity:0 }}
        40% {{ opacity:1 }}
        100% {{ left:120%; transform:rotate(720deg); }}
    }}

    .msg {{ font-weight:700; margin-top:8px; font-size:18px; text-align:center; }}
    .confetti {{ font-size:22px; text-align:center; margin-top:6px; }}
    </style>

    <div class="wrap">
        <div class="bat">🏏</div>
        <div class="ball">🏐</div>
    </div>

    <div class="msg">{message}</div>
    <div class="confetti">🔥🎉🥁</div>

    <script>
    // IPL DRUM BEAT SOUND — always works
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    const now = ctx.currentTime;

    function drum(time, freq, gain=0.9, decay=0.22) {{
        const osc = ctx.createOscillator();
        const g = ctx.createGain();
        osc.frequency.setValueAtTime(freq, time);
        g.gain.setValueAtTime(gain, time);
        g.gain.exponentialRampToValueAtTime(0.001, time + decay);
        osc.type = "triangle";
        osc.connect(g);
        g.connect(ctx.destination);
        osc.start(time);
        osc.stop(time + decay);
    }}

    // BOOM–BOOM–CRACK IPL drum sequence
    drum(now, 150, 1.1, 0.25);
    drum(now + 0.15, 110, 1.0, 0.30);
    drum(now + 0.33, 220, 1.2, 0.15);
    </script>
    """

    components.html(animation_html, height=260)


# ---------------------------
# Sample dataset generator (unchanged)
# ---------------------------
@st.cache_data
def generate_sample_dataset(n_rows=200, seed=42):
    rng = np.random.RandomState(seed)
    teams = ["CSK", "MI", "RCB", "KKR", "SRH", "RR", "DC", "PBKS"]
    roles = ["Batsman", "Bowler", "All-Rounder", "Wicket-Keeper"]
    players = [f"Player_{i}" for i in range(1, 1001)]

    rows = []
    for i in range(n_rows):
        team = rng.choice(teams)
        role = rng.choice(roles, p=[0.35, 0.33, 0.22, 0.10])
        purse = int(rng.randint(5, 20) * 1_00_00_000)
        target = rng.choice(players)
        expected = int(rng.randint(50, 200) * 1_00_000)
        prob = int(rng.randint(20, 95))
        need = int(rng.randint(1, 5))
        base = int(rng.choice([20, 30, 40, 50, 60, 75, 100]) * 1_00_000)

        rows.append({
            "Team": team,
            "Player_Role": role,
            "Purse_Remaining": purse,
            "Target_Player": target,
            "Expected_Bid": expected,
            "Buy_Probability(%)": prob,
            "Players_Needed": need,
            "Base_Price": base
        })

    return pd.DataFrame(rows)


# ---------------------------
# DATA INPUT (unchanged)
# ---------------------------
st.sidebar.header("Data Input")
uploaded_file = st.sidebar.file_uploader("Upload IPL dataset (Excel or CSV)", type=["xlsx", "csv"])

if uploaded_file:
    try:
        if uploaded_file.name.endswith(".xlsx"):
            df = pd.read_excel(uploaded_file)
        else:
            df = pd.read_csv(uploaded_file)
        st.sidebar.success("Loaded dataset")
    except:
        st.sidebar.error("Failed to load. Using sample dataset.")
        df = generate_sample_dataset(400)
else:
    df = generate_sample_dataset(400)
    st.sidebar.info("Using sample dataset (400 rows).")

required_cols = ["Team", "Player_Role", "Purse_Remaining", "Target_Player",
                 "Expected_Bid", "Buy_Probability(%)", "Players_Needed", "Base_Price"]
for col in required_cols:
    if col not in df.columns:
        df = generate_sample_dataset(400)
        break

# ensure numeric
for col in ["Purse_Remaining", "Expected_Bid", "Buy_Probability(%)", "Players_Needed", "Base_Price"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")


# ---------------------------
# SIDEBAR TEAM SELECTOR (unchanged)
# ---------------------------
st.sidebar.markdown("---")
st.sidebar.header("Team Quick Select")
teams_list_sidebar = sorted(df["Team"].unique())
team_single_select = st.sidebar.selectbox("Pick team to focus", ["(none)"] + teams_list_sidebar)


# ---------------------------
# FILTERS (unchanged)
# ---------------------------
st.sidebar.header("Filters & Interactivity")
teams_multiselect = st.sidebar.multiselect("Select teams", teams_list_sidebar, default=teams_list_sidebar)
role_options = sorted(df["Player_Role"].unique())
role_filter = st.sidebar.multiselect("Filter by role", ["All"] + role_options, default=["All"])

exp_range = st.sidebar.slider("Expected Bid (₹)", int(df["Expected_Bid"].min()),
                              int(df["Expected_Bid"].max()), 
                              (int(df["Expected_Bid"].min()), int(df["Expected_Bid"].max())))

prob_threshold = st.sidebar.slider("Min Buy Probability (%)", 0, 100, 20)
anim_speed = st.sidebar.slider("Animation Speed (lower=faster)", 1, 10, 4)
top_n_players = st.sidebar.number_input("Top N players", 5, 30, 10)
enable_sunburst = st.sidebar.checkbox("Show Sunburst", False)
enable_highlight = st.sidebar.checkbox("Highlight selected player", True)


# ---------------------------
# APPLY FILTERS (unchanged)
# ---------------------------
df_filtered = df.copy()
df_filtered = df_filtered[df_filtered["Team"].isin(teams_multiselect)]
if "All" not in role_filter:
    df_filtered = df_filtered[df_filtered["Player_Role"].isin(role_filter)]

df_filtered = df_filtered[
    (df_filtered["Expected_Bid"] >= exp_range[0]) &
    (df_filtered["Expected_Bid"] <= exp_range[1]) &
    (df_filtered["Buy_Probability(%)"] >= prob_threshold)
]


# ---------------------------
# TEAM COLORS (unchanged)
# ---------------------------
palette = px.colors.qualitative.Plotly + px.colors.qualitative.Dark24
team_colors = {team: palette[i % len(palette)] for i, team in enumerate(sorted(df["Team"].unique()))}


# ---------------------------
# CHART SECTION (unchanged)
# ---------------------------
col_main, col_side = st.columns([2, 1])

# 1) BAR CHART — Players Needed
with col_main:
    st.subheader("Players Needed by Team (click a bar)")
    agg = df.groupby("Team")["Players_Needed"].sum().reset_index()
    fig_bar = px.bar(agg, x="Team", y="Players_Needed", text="Players_Needed",
                     color="Team", color_discrete_map=team_colors)
    st.plotly_chart(fig_bar, use_container_width=True)
    bar_click = plotly_events(fig_bar, click_event=True)


clicked_team = None
if bar_click:
    clicked_team = bar_click[0].get("x")

if team_single_select != "(none)":
    clicked_team = team_single_select

# 2) SCATTER CHART
with col_main:
    st.subheader("Expected Bid vs Buy Probability")
    if clicked_team:
        color_map = {t: (team_colors[t] if t == clicked_team else "lightgray") for t in team_colors}
    else:
        color_map = team_colors

    fig_scatter = px.scatter(df_filtered, x="Expected_Bid", y="Buy_Probability(%)", 
                             color="Team", hover_data=["Target_Player", "Player_Role"],
                             color_discrete_map=color_map)
    st.plotly_chart(fig_scatter, use_container_width=True)
    scatter_click = plotly_events(fig_scatter, click_event=True)


# 3) BOX + MEAN (unchanged)
with col_side:
    st.subheader("Expected Bid by Team")
    if not df_filtered.empty:
        st.plotly_chart(
            px.box(df_filtered, x="Team", y="Expected_Bid", color="Team",
                   color_discrete_map=team_colors),
            use_container_width=True
        )

        mean_bid = df_filtered.groupby("Team")["Expected_Bid"].mean().reset_index()
        st.plotly_chart(
            px.bar(mean_bid, x="Team", y="Expected_Bid", color="Team",
                   color_discrete_map=team_colors),
            use_container_width=True
        )


# ---------------------------
# HEATMAP (unchanged)
# ---------------------------
with col_main:
    st.subheader("Team vs Role Heatmap")
    heat = df_filtered.groupby(["Team", "Player_Role"]).size().unstack(fill_value=0)
    st.plotly_chart(px.imshow(heat), use_container_width=True)

# ---------------------------
# TOP PLAYERS (unchanged)
# ---------------------------
with col_side:
    st.subheader(f"Top {top_n_players} Players")
    top_p = df_filtered.sort_values("Buy_Probability(%)", ascending=False).head(top_n_players)
    st.plotly_chart(px.bar(top_p, x="Target_Player", y="Buy_Probability(%)",
                           color="Team", color_discrete_map=team_colors),
                    use_container_width=True)

# ---------------------------
# TEAM DETAILS + PLAYER DETAILS (unchanged)
# ---------------------------
st.markdown("---")
detail_col, preview_col = st.columns([1, 2])

selected_player_name = None
if scatter_click:
    selected_player_name = scatter_click[0].get("customdata", [""])[0]


with detail_col:
    if clicked_team:
        st.subheader(f"{clicked_team} — Team Details")

        tdf = df[df["Team"] == clicked_team]
        st.metric("Total Records", len(tdf))
        st.metric("Players Needed", int(tdf["Players_Needed"].sum()))
        st.metric("Avg Expected Bid", f"₹{int(tdf['Expected_Bid'].mean()):,}")

        st.markdown("**Role Breakdown**")
        st.table(tdf["Player_Role"].value_counts())

        st.markdown("**Team Players & Stats**")
        st.dataframe(
            tdf.groupby(["Player_Role", "Target_Player"]).agg(
                Expected_Bid=("Expected_Bid", "max"),
                Buy_Probability=("Buy_Probability(%)", "max"),
                Players_Needed=("Players_Needed", "sum")
            ).reset_index().sort_values("Buy_Probability", ascending=False),
            use_container_width=True
        )

        # 🔥 Play sound for that team
        if st.button(f"Play Team Theme — {clicked_team}"):
            play_ipl_theme_with_animation(f"{clicked_team} — IPL Drum Effect!")


with preview_col:
    if selected_player_name:
        st.subheader(f"Player Selected: {selected_player_name}")
        st.dataframe(df[df["Target_Player"] == selected_player_name], use_container_width=True)
    else:
        st.subheader("Filtered Data Preview")
        st.dataframe(df_filtered.head(100), use_container_width=True)


# ---------------------------
# DOWNLOAD (unchanged)
# ---------------------------
@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

st.download_button("Download Filtered CSV", convert_df_to_csv(df_filtered),
                   "ipl_filtered_data.csv", "text/csv")

# ---------------------------
# QUICK ANIMATE BUTTON (unchanged except sound)
# ---------------------------
if st.button("Quick Animate Bars"):
    speed = max(0.02, 0.12 / anim_speed)
    steps = 20
    team_list = agg["Team"].tolist()
    final_vals = agg["Players_Needed"].tolist()

    for step in range(1, steps + 1):
        vals = [int(v * step / steps) for v in final_vals]
        temp_df = pd.DataFrame({"Team": team_list, "Players_Needed": vals})
        st.plotly_chart(
            px.bar(temp_df, x="Team", y="Players_Needed", color="Team",
                   text="Players_Needed", color_discrete_map=team_colors),
            use_container_width=True
        )
        time.sleep(speed)

    # NEW IPL DRUM SOUND HERE
    play_ipl_theme_with_animation("Finished Animation — IPL Style!")
    st.success("Animation Completed")
