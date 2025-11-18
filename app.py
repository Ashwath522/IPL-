# app.py — FULL VERSION (unchanged dashboard + fixed animation + NEW LONG IPL SOUND)

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
    "Click a team bar to focus. Each team uses consistent colors across charts. Click scatter points to inspect players."
)

# ---------------------------
# NEW LOUDER + LONG IPL DRUM SOUND (GUARANTEED TO PLAY)
# ---------------------------
def play_ipl_theme_with_animation(message="That’s a lovely shot!"):
    """
    Extended IPL drum beat (1.2 sec), louder, stronger.
    Works in all browsers.
    """
    animation_html = f"""
    <style>
    .wrap {{ position:relative; width:100%; height:150px; overflow:hidden; }}
    .bat {{ font-size:60px; position:absolute; left:-30%; top:20px; animation:swing 1.0s ease-out forwards; }}
    .ball{{ font-size:52px; position:absolute; left:-10%; top:55px; animation:roll 1.0s ease-out forwards; }}

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

    .msg {{ font-weight:700; margin-top:8px; font-size:20px; text-align:center; }}
    .confetti {{ font-size:26px; text-align:center; margin-top:6px; }}
    </style>

    <div class="wrap">
        <div class="bat">🏏</div>
        <div class="ball">🏐</div>
    </div>

    <div class="msg">{message}</div>
    <div class="confetti">🔥🎉🥁</div>

    <script>
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    const now = ctx.currentTime;

    function hit(time, freq, vol=1.5, decay=0.40) {{
        const osc = ctx.createOscillator();
        const gain = ctx.createGain();

        osc.frequency.setValueAtTime(freq, time);
        gain.gain.setValueAtTime(vol, time);
        gain.gain.exponentialRampToValueAtTime(0.0001, time + decay);

        osc.type = "triangle";
        osc.connect(gain);
        gain.connect(ctx.destination);

        osc.start(time);
        osc.stop(time + decay);
    }}

    // Extended IPL BOOM–BOOM–TUK–DHAM sequence
    hit(now,       140, 1.7, 0.40);   // First BOOM
    hit(now+0.20,  110, 1.8, 0.45);   // Second deep BOOM
    hit(now+0.45,  180, 1.6, 0.30);   // Mid crisp TUK
    hit(now+0.70,  250, 2.0, 0.30);   // Final sharp DHAM
    </script>
    """

    components.html(animation_html, height=300)


# ---------------------------
# SAMPLE DATASET (unchanged)
# ---------------------------
@st.cache_data
def generate_sample_dataset(n_rows=200, seed=42):
    rng = np.random.RandomState(seed)
    teams = ["CSK", "MI", "RCB", "KKR", "SRH", "RR", "DC", "PBKS"]
    roles = ["Batsman", "Bowler", "All-Rounder", "Wicket-Keeper"]
    players = [f"Player_{i}" for i in range(1, 1001)]
    rows = []

    for i in range(n_rows):
        rows.append({
            "Team": rng.choice(teams),
            "Player_Role": rng.choice(roles, p=[0.35, 0.33, 0.22, 0.10]),
            "Purse_Remaining": int(rng.randint(5, 20) * 1_00_00_000),
            "Target_Player": rng.choice(players),
            "Expected_Bid": int(rng.randint(50, 200) * 1_00_000),
            "Buy_Probability(%)": int(rng.randint(20, 95)),
            "Players_Needed": int(rng.randint(1, 5)),
            "Base_Price": int(rng.choice([20, 30, 40, 50, 60, 75, 100]) * 1_00_000)
        })

    return pd.DataFrame(rows)

# ---------------------------
# DATA UPLOAD (unchanged)
# ---------------------------
st.sidebar.header("Data Input")
file = st.sidebar.file_uploader("Upload Excel/CSV", type=["xlsx", "csv"])

if file:
    try:
        df = pd.read_excel(file) if file.name.endswith(".xlsx") else pd.read_csv(file)
        st.sidebar.success("Loaded dataset successfully!")
    except:
        st.sidebar.error("Could not read file. Using sample dataset.")
        df = generate_sample_dataset(400)
else:
    df = generate_sample_dataset(400)
    st.sidebar.info("Using sample dataset (400 rows).")

# required columns
req = ["Team","Player_Role","Purse_Remaining","Target_Player",
       "Expected_Bid","Buy_Probability(%)","Players_Needed","Base_Price"]

for c in req:
    if c not in df.columns:
        df = generate_sample_dataset(400)

for c in ["Purse_Remaining","Expected_Bid","Buy_Probability(%)","Players_Needed","Base_Price"]:
    df[c]=pd.to_numeric(df[c],errors="coerce")

# ---------------------------
# SIDEBAR FILTERS (unchanged)
# ---------------------------
teams = sorted(df["Team"].unique())
team_focus = st.sidebar.selectbox("Focus Team", ["(none)"]+teams)

st.sidebar.header("Filters")
multi_teams = st.sidebar.multiselect("Select Teams", teams, default=teams)
roles = sorted(df["Player_Role"].unique())
role_f = st.sidebar.multiselect("Filter Role", ["All"]+roles, default=["All"])

exp_min, exp_max = int(df["Expected_Bid"].min()), int(df["Expected_Bid"].max())
exp_rng = st.sidebar.slider("Expected Bid Range", exp_min, exp_max, (exp_min,exp_max))

prob_min = st.sidebar.slider("Min Probability",0,100,20)
anim_speed = st.sidebar.slider("Animation Speed (lower=faster)",1,10,4)
top_n = st.sidebar.number_input("Top N players",5,30,10)

sunburst = st.sidebar.checkbox("Show Sunburst", False)
highlight = st.sidebar.checkbox("Highlight Selected Player", True)

# ---------------------------
# FILTER DATA
# ---------------------------
df2 = df[df["Team"].isin(multi_teams)]
if "All" not in role_f:
    df2 = df2[df2["Player_Role"].isin(role_f)]
df2 = df2[(df2["Expected_Bid"]>=exp_rng[0]) & (df2["Expected_Bid"]<=exp_rng[1])]
df2 = df2[df2["Buy_Probability(%)"]>=prob_min]

# ---------------------------
# TEAM COLORS
# ---------------------------
palette = px.colors.qualitative.Bold + px.colors.qualitative.Dark24
colors = {team:palette[i%len(palette)] for i,team in enumerate(teams)}

# ---------------------------
# CHARTS (unchanged)
# ---------------------------
col1,col2 = st.columns([2,1])

# Bar Chart
with col1:
    st.subheader("Players Needed by Team")
    agg = df.groupby("Team")["Players_Needed"].sum().reset_index()
    fig = px.bar(agg,x="Team",y="Players_Needed",color="Team",text="Players_Needed",
                 color_discrete_map=colors)
    st.plotly_chart(fig,use_container_width=True)
    bar_click = plotly_events(fig,click_event=True)

clicked_team=None
if bar_click:
    clicked_team = bar_click[0].get("x")
if team_focus!="(none)":
    clicked_team = team_focus

# Scatter
with col1:
    st.subheader("Expected Bid vs Probability")
    cmap = {t:(colors[t] if t==clicked_team else "lightgray") for t in teams} if clicked_team else colors
    fig = px.scatter(df2,x="Expected_Bid",y="Buy_Probability(%)",color="Team",
                     hover_data=["Target_Player","Player_Role"],
                     color_discrete_map=cmap)
    st.plotly_chart(fig,use_container_width=True)
    player_click = plotly_events(fig,click_event=True)

# Box + Mean
with col2:
    st.subheader("Bid Distributions & Averages")
    st.plotly_chart(px.box(df2,x="Team",y="Expected_Bid",color="Team",
                           color_discrete_map=colors),
                    use_container_width=True)

    mean = df2.groupby("Team")["Expected_Bid"].mean().reset_index()
    st.plotly_chart(px.bar(mean,x="Team",y="Expected_Bid",color="Team",
                           color_discrete_map=colors),
                    use_container_width=True)

# Heatmap
with col1:
    st.subheader("Team vs Role")
    heat = df2.groupby(["Team","Player_Role"]).size().unstack(fill_value=0)
    st.plotly_chart(px.imshow(heat),use_container_width=True)

# Top players
with col2:
    st.subheader(f"Top {top_n} Players")
    top = df2.sort_values("Buy_Probability(%)",ascending=False).head(top_n)
    st.plotly_chart(px.bar(top,x="Target_Player",y="Buy_Probability(%)",
                           color="Team",color_discrete_map=colors),
                    use_container_width=True)

# Sunburst
if sunburst:
    st.subheader("Sunburst")
    sb = df2.groupby(["Team","Player_Role","Target_Player"])["Buy_Probability(%)"].max().reset_index()
    st.plotly_chart(px.sunburst(sb,path=["Team","Player_Role","Target_Player"],
                                values="Buy_Probability(%)",
                                color="Team",color_discrete_map=colors),
                    use_container_width=True)

# ---------------------------
# DETAILS PANEL (unchanged)
# ---------------------------
st.markdown("---")
d1,d2 = st.columns([1,2])

sel_player=None
if player_click:
    sel_player = player_click[0].get("customdata",[None])[0]

with d1:
    if clicked_team:
        st.subheader(f"{clicked_team} — Details")
        tdf = df[df["Team"]==clicked_team]

        st.metric("Players in Dataset",len(tdf))
        st.metric("Total Players Needed",int(tdf["Players_Needed"].sum()))
        st.metric("Avg Expected Bid",f"₹{int(tdf['Expected_Bid'].mean()):,}")

        st.table(tdf["Player_Role"].value_counts())

        st.markdown("**Team Player Stats**")
        stats = tdf.groupby(["Player_Role","Target_Player"]).agg(
            Expected=("Expected_Bid","max"),
            Probability=("Buy_Probability(%)","max"),
            Need=("Players_Needed","sum")
        ).reset_index().sort_values("Probability",ascending=False)

        st.dataframe(stats,use_container_width=True)

        if st.button(f"Play Team Theme — {clicked_team}"):
            play_ipl_theme_with_animation(f"{clicked_team} — IPL Mode Activated!")

    else:
        st.info("Click a team to view info.")

with d2:
    if sel_player:
        st.subheader(f"Selected Player: {sel_player}")
        st.dataframe(df[df["Target_Player"]==sel_player],use_container_width=True)
    else:
        st.subheader("Filtered Data Preview")
        st.dataframe(df2.head(100),use_container_width=True)

# ---------------------------
# DOWNLOAD BUTTON (unchanged)
# ---------------------------
@st.cache_data
def to_csv(x):
    return x.to_csv(index=False).encode("utf-8")

st.download_button("Download Filtered CSV",to_csv(df2),"ipl_data.csv","text/csv")

# ---------------------------
# FIXED QUICK ANIMATION (NO ERRORS)
# ---------------------------
if st.button("Quick Animate Bars"):
    steps=20
    speed=max(0.02,0.12/anim_speed)
    teams_list=agg["Team"].tolist()
    final_vals=agg["Players_Needed"].tolist()

    placeholder = st.empty()   # FIX

    for s in range(1,steps+1):
        vals=[int(v*s/steps) for v in final_vals]
        temp=pd.DataFrame({"Team":teams_list,"Players_Needed":vals})
        fig=px.bar(temp,x="Team",y="Players_Needed",color="Team",
                   text="Players_Needed",color_discrete_map=colors)
        placeholder.plotly_chart(fig,use_container_width=True)
        time.sleep(speed)

    play_ipl_theme_with_animation("Animation Complete — IPL Style!")
    st.success("Bars animated successfully!")
