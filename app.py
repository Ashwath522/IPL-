# app.py — Version A merged with per-team selection + IPL theme (local path)
# Keep all original charts/filters/behavior; added team select and Play Team Theme (plays local mp3).
import os
import time
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from streamlit_plotly_events import plotly_events
import streamlit.components.v1 as components

# ---------------------------
# Config
# ---------------------------
st.set_page_config(layout="wide", page_title="IPL Auction Visualizer (Interactive)", page_icon="🏏")

st.title("IPL Auction Visualizer — Interactive & Focused")
st.markdown(
    "Reduced to 5-6 interactive charts. Click a team bar to focus on a team (or pick one team in the sidebar). "
    "Each team uses a consistent color across charts. Click scatter points to inspect players."
)

# ---------------------------
# Your local MP3 path (use this exact path on your local machine)
# ---------------------------
THEME_MP3_PATH = r"d:\Users\DELL\Downloads\ipl_theme.mp3"

# ---------------------------
# Animation + MP3 player (uses local path if found)
# ---------------------------
def play_ipl_theme_with_animation(message="That’s a lovely shot!"):
    """
    Shows a short bat+ball animation and plays the MP3 at THEME_MP3_PATH (local file).
    If the mp3 path does not exist, shows a fallback short synthetic audio + animation.
    """
    mp3_path = THEME_MP3_PATH.replace("\\", "/")
    if os.path.exists(THEME_MP3_PATH):
        html = f"""
        <style>
        .wrap {{ position:relative; width:100%; height:140px; overflow:hidden; }}
        .bat {{ font-size:52px; position:absolute; left:-30%; top:10px; animation:swing 0.8s ease-out 0s 1 forwards; }}
        .ball{{ font-size:48px; position:absolute; left:-10%; top:40px; animation:roll 0.8s ease-out 0s 1 forwards; }}
        @keyframes swing {{ 0%{{left:-30%; transform:rotate(-30deg); opacity:0}} 30%{{left:8%; opacity:1}} 100%{{left:110%; transform:rotate(20deg);}} }}
        @keyframes roll  {{ 0%{{left:-10%; transform:rotate(0deg); opacity:0}} 40%{{opacity:1}} 100%{{left:120%; transform:rotate(720deg);}} }}
        .msg {{ font-weight:700; margin-top:6px; font-size:16px; text-align:center; }}
        .confetti {{ font-size:22px; text-align:center; margin-top:6px; }}
        </style>

        <div class="wrap"><div class="bat">🏏</div><div class="ball">🏐</div></div>
        <div class="msg">{message}</div>
        <div class="confetti">🎉👏🔥</div>

        <audio autoplay>
          <source src="file:///{mp3_path}" type="audio/mpeg">
          Your browser does not support the audio element.
        </audio>
        """
        # height should allow player UI and visuals
        components.html(html, height=240)
    else:
        # fallback to synthetic short audio + animation if mp3 not found
        js_html = r"""
        <style>
        .celebrate { text-align:center; font-weight:700; margin-top:6px; }
        .wrap { position:relative; width:100%; height:90px; overflow:hidden; }
        .bat { font-size:48px; position:absolute; left:-30%; top:8px; animation: swing 0.7s ease-out 0s 1 forwards; }
        .ball{ font-size:44px; position:absolute; left:-10%; top:30px; animation: roll 0.7s ease-out 0s 1 forwards; }
        @keyframes swing { 0%{left:-30%; transform:rotate(-30deg); opacity:0} 30%{left:8%; opacity:1} 100%{left:110%; transform:rotate(20deg);} }
        @keyframes roll  { 0%{left:-10%; transform:rotate(0deg); opacity:0} 40%{opacity:1} 100%{left:120%; transform:rotate(720deg);} }
        .confetti { font-size:22px; margin-top:6px; }
        </style>
        <div class="wrap"><div class="bat">🏏</div><div class="ball">🏐</div></div>
        <div class="celebrate">""" + message + r"""</div>
        <div class="confetti">🎉👏🔥</div>

        <script>
        try {
          const ctx = new (window.AudioContext || window.webkitAudioContext)();
          const now = ctx.currentTime;
          function playBlast(time, duration, gainVal, freq) {
            const o = ctx.createOscillator();
            const g = ctx.createGain();
            o.type = 'square';
            o.frequency.setValueAtTime(freq, time);
            g.gain.setValueAtTime(gainVal, time);
            g.gain.exponentialRampToValueAtTime(0.001, time + duration);
            o.connect(g); g.connect(ctx.destination);
            o.start(time);
            o.stop(time + duration + 0.02);
          }
          let t = now + 0.05;
          const short = 0.06;
          playBlast(t, short, 0.04, 600); t += 0.09;
          playBlast(t, short, 0.04, 700); t += 0.09;
          playBlast(t, short, 0.04, 800); t += 0.2;
          playBlast(t, short, 0.045, 600); t += 0.09;
          playBlast(t, short, 0.045, 700); t += 0.09;
          playBlast(t, short, 0.045, 800);
        } catch(e) {
          console.log("Audio error:", e);
        }
        </script>
        """
        components.html(js_html, height=170)

# ---------------------------
# Sample dataset generator (original)
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
    df = pd.DataFrame(rows)
    return df

# ---------------------------
# Sidebar: dataset upload or use sample
# ---------------------------
st.sidebar.header("Data Input")
uploaded_file = st.sidebar.file_uploader("Upload IPL dataset (Excel or CSV). Leave empty to use sample dataset.", type=["xlsx", "csv"])

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith(".xlsx"):
            df = pd.read_excel(uploaded_file)
        else:
            df = pd.read_csv(uploaded_file)
        st.sidebar.success("Loaded dataset: " + uploaded_file.name)
    except Exception as e:
        st.sidebar.error("Could not read file. Using sample dataset. Error: " + str(e))
        df = generate_sample_dataset(400)
else:
    df = generate_sample_dataset(400)
    st.sidebar.info("Using sample dataset (400 rows).")

required_cols = ["Team", "Player_Role", "Purse_Remaining", "Target_Player", "Expected_Bid", "Buy_Probability(%)", "Players_Needed", "Base_Price"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    st.warning(f"Uploaded dataset is missing columns: {missing}. The app expects columns: {required_cols}. Using sample dataset instead.")
    df = generate_sample_dataset(400)

# ensure numeric columns
for col in ["Purse_Remaining", "Expected_Bid", "Buy_Probability(%)", "Players_Needed", "Base_Price"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# ---------------------------
# NEW: Sidebar single-team selector (user requested)
# ---------------------------
st.sidebar.markdown("---")
st.sidebar.header("Team Quick Select")
teams_list_sidebar = sorted(df["Team"].dropna().unique())
team_single_select = st.sidebar.selectbox("Pick one team to focus (optional)", options=["(none)"] + teams_list_sidebar, index=0)

# Sidebar filters & interaction settings (existing)
st.sidebar.header("Filters & Interactivity")
teams_list = sorted(df["Team"].dropna().unique())
teams_multiselect = st.sidebar.multiselect("Select teams (multi)", options=teams_list, default=teams_list)
role_options = sorted(df["Player_Role"].dropna().unique())
role_filter = st.sidebar.multiselect("Filter by role", options=["All"] + role_options, default=["All"])
exp_min = int(df["Expected_Bid"].min()) if not df["Expected_Bid"].isnull().all() else 0
exp_max = int(df["Expected_Bid"].max()) if not df["Expected_Bid"].isnull().all() else 1_00_00_000
exp_range = st.sidebar.slider("Expected Bid range (₹)", exp_min, exp_max, (exp_min, exp_max))
prob_threshold = st.sidebar.slider("Min Buy Probability (%)", 0, 100, 20)
anim_speed = st.sidebar.slider("Animation speed (lower = faster)", 1, 10, 4)
top_n_players = st.sidebar.number_input("Top N players to show", min_value=5, max_value=30, value=10, step=1)
st.sidebar.markdown("---")
enable_sunburst = st.sidebar.checkbox("Show Team→Role→Player sunburst (optional)", value=False)
enable_highlight = st.sidebar.checkbox("Highlight selected player in scatter/table", value=True)

# Apply filters
df_filtered = df.copy()
if teams_multiselect:
    df_filtered = df_filtered[df_filtered["Team"].isin(teams_multiselect)]
if "All" not in role_filter:
    df_filtered = df_filtered[df_filtered["Player_Role"].isin(role_filter)]
df_filtered = df_filtered[(df_filtered["Expected_Bid"] >= exp_range[0]) & (df_filtered["Expected_Bid"] <= exp_range[1])]
df_filtered = df_filtered[(df_filtered["Buy_Probability(%)"] >= prob_threshold)]

# Generate a consistent color map per team (cycles through a qualitative palette)
palette = px.colors.qualitative.Plotly + px.colors.qualitative.Dark24 + px.colors.qualitative.Bold
unique_teams = sorted(df["Team"].dropna().unique())
team_colors = {team: palette[i % len(palette)] for i, team in enumerate(unique_teams)}

# Layout: keep about 5-6 charts
col_main, col_side = st.columns([2, 1])

# 1) Main bar chart: Players Needed by Team (clickable)
with col_main:
    st.subheader("Players Needed by Team (click a bar to focus)")
    agg = df.groupby("Team")["Players_Needed"].sum().reset_index().sort_values("Players_Needed", ascending=False)
    fig_bar = px.bar(agg, x="Team", y="Players_Needed", text="Players_Needed",
                     title="Total Players Needed by Team",
                     color="Team", color_discrete_map=team_colors)
    fig_bar.update_traces(marker_line_color='black', marker_line_width=1)
    st.plotly_chart(fig_bar, use_container_width=True)
    # capture click on bar
    bar_click = plotly_events(fig_bar, click_event=True, hover_event=False)

clicked_team = None
if bar_click:
    bp = bar_click[0]
    clicked_team = bp.get("x") or bp.get("label") or None

# If user forced a single team selection in sidebar, prefer that
if team_single_select and team_single_select != "(none)":
    clicked_team = team_single_select
elif len(teams_multiselect) == 1:
    clicked_team = teams_multiselect[0]

# 2) Scatter: Expected Bid vs Buy Probability (click a point to pick player)
with col_main:
    st.subheader("Expected Bid vs Buy Probability")
    if not df_filtered.empty:
        # Highlight chosen team by using its color and dim others slightly
        if clicked_team:
            color_map = {t: team_colors[t] if t == clicked_team else "lightgray" for t in unique_teams}
        else:
            color_map = team_colors
        fig_scatter = px.scatter(df_filtered, x="Expected_Bid", y="Buy_Probability(%)", color="Team",
                                 hover_data=["Target_Player", "Player_Role", "Base_Price"],
                                 title="Expected Bid vs Buy Probability",
                                 color_discrete_map=color_map,
                                 height=420)
        fig_scatter.update_layout(clickmode="event+select")
        st.plotly_chart(fig_scatter, use_container_width=True)
        scatter_click = plotly_events(fig_scatter, click_event=True, hover_event=False)
    else:
        st.info("No data for current filters.")
        scatter_click = []

# 3) Box + Mean bar: Expected Bid by Team (compact)
with col_side:
    st.subheader("Expected Bid by Team")
    if not df_filtered.empty:
        # boxplot keeps consistent team colors
        fig_box = px.box(df_filtered, x="Team", y="Expected_Bid", color="Team", color_discrete_map=team_colors, points="outliers")
        st.plotly_chart(fig_box, use_container_width=True)
        # mean bar
        mean_by_team = df_filtered.groupby("Team")["Expected_Bid"].mean().reset_index().sort_values("Expected_Bid", ascending=False)
        fig_mean = px.bar(mean_by_team, x="Team", y="Expected_Bid", text="Expected_Bid", color="Team", color_discrete_map=team_colors)
        fig_mean.update_traces(texttemplate="₹%{text:,.0f}", textposition="outside")
        st.plotly_chart(fig_mean, use_container_width=True)
    else:
        st.info("No data to display Expected Bid charts")

# 4) Heatmap: Team vs Role counts
with col_main:
    st.subheader("Team vs Role (heatmap)")
    if not df_filtered.empty:
        heat = df_filtered.groupby(["Team", "Player_Role"]).size().unstack(fill_value=0)
        fig_heat = px.imshow(heat, labels=dict(x="Player Role", y="Team", color="Count"), x=heat.columns, y=heat.index, aspect="auto", title="Players count by Team and Role")
        # apply team colors to y-axis by mapping tickfont color (plotly doesn't support per-tick color in a simple call, but we'll set tickfont color uniformly)
        st.plotly_chart(fig_heat, use_container_width=True)
    else:
        st.info("No data for heatmap")

# 5) Top players (compact bar)
with col_side:
    st.subheader(f"Top {top_n_players} Target Players")
    if not df_filtered.empty:
        top_players = (
            df_filtered.groupby("Target_Player")
            .agg({"Buy_Probability(%)": "max", "Expected_Bid": "max", "Team": lambda x: x.mode().iat[0] if not x.mode().empty else x.iloc[0]})
            .reset_index()
            .sort_values(["Buy_Probability(%)", "Expected_Bid"], ascending=[False, False])
            .head(int(top_n_players))
        )
        # color by team with the same consistent map
        fig_top = px.bar(top_players, x="Target_Player", y="Buy_Probability(%)", color="Team",
                         color_discrete_map=team_colors, hover_data=["Expected_Bid"], title="Top Target Players — Buy Probability")
        fig_top.update_layout(xaxis_tickangle=-45, height=380)
        st.plotly_chart(fig_top, use_container_width=True)
    else:
        st.info("No data for top players")

# Optional 6) Sunburst (toggle)
if enable_sunburst and not df_filtered.empty:
    st.subheader("Team → Role → Player (sunburst)")
    top_entries = (df_filtered.groupby(["Team", "Player_Role", "Target_Player"])
                   .agg({"Buy_Probability(%)": "max"})
                   .reset_index())
    top_entries = top_entries.sort_values("Buy_Probability(%)", ascending=False).head(200)
    fig_sun = px.sunburst(top_entries, path=["Team", "Player_Role", "Target_Player"], values="Buy_Probability(%)",
                          color="Team", color_discrete_map=team_colors, title="Sunburst: Team -> Role -> Player")
    st.plotly_chart(fig_sun, use_container_width=True)

# Panel: show details for clicked team or selected player
st.markdown("---")
detail_col, preview_col = st.columns([1, 2])

# Determine selected player (from scatter click)
selected_player_name = None
if scatter_click:
    sp = scatter_click[0]
    # Try to extract the player name from customdata or hovertext
    cd = sp.get("customdata")
    hovertext = sp.get("hovertext")
    txt = sp.get("text")
    # Our hover_data included "Target_Player" so customdata may contain it
    if isinstance(cd, (list, tuple)) and cd:
        # find a string that doesn't look like numeric
        for item in cd:
            if isinstance(item, str) and len(item) > 2:
                selected_player_name = item
                break
    if selected_player_name is None and isinstance(hovertext, str):
        selected_player_name = hovertext
    if selected_player_name is None and isinstance(txt, str):
        selected_player_name = txt
    # fallback: try matching by x,y coordinates (Expected_Bid/Buy_Probability)
    if selected_player_name is None:
        x_val = sp.get("x")
        y_val = sp.get("y")
        if x_val is not None and y_val is not None:
            close = df_filtered[
                (np.isclose(df_filtered["Expected_Bid"], float(x_val), atol=1e-6)) &
                (np.isclose(df_filtered["Buy_Probability(%)"], float(y_val), atol=1e-6))
            ]
            if not close.empty:
                selected_player_name = close.iloc[0]["Target_Player"]

# Show team metrics on the left
with detail_col:
    if clicked_team:
        st.subheader(f"Team: {clicked_team}")
        team_df = df[df["Team"] == clicked_team]
        if not team_df.empty:
            st.metric("Players in dataset", len(team_df))
            st.metric("Sum Players Needed", int(team_df["Players_Needed"].sum()))
            st.metric("Avg Expected Bid", f"₹{int(team_df['Expected_Bid'].mean()):,}")
            # small role breakdown
            rcounts = team_df["Player_Role"].value_counts().reset_index()
            rcounts.columns = ["Player_Role", "Count"]
            st.table(rcounts)

            # Also show detailed team stats column similar to previous ask (players list & counts)
            st.markdown("**Team Players & Needs**")
            team_stats = team_df.groupby(["Player_Role", "Target_Player"]).agg(
                Expected_Bid=("Expected_Bid", "max"),
                Buy_Probability=("Buy_Probability(%)", "max"),
                Players_Needed=("Players_Needed", "sum")
            ).reset_index().sort_values(["Buy_Probability", "Expected_Bid"], ascending=[False, False])
            st.dataframe(team_stats.head(50), use_container_width=True)

            # New: Play team theme button — plays the IPL MP3 (local path) + animation for this team
            if st.button(f"Play Team Theme — {clicked_team}"):
                # play the mp3 + animation with team message
                play_ipl_theme_with_animation(f"{clicked_team} — Enjoy the team theme!")
        else:
            st.info("No records for this team.")
    else:
        st.subheader("Overview")
        st.markdown("Click a team bar or select a single team in the sidebar to view team details.")

# Show selected player / preview table on the right
with preview_col:
    if selected_player_name and enable_highlight:
        st.subheader(f"Selected Player: {selected_player_name}")
        player_rows = df[df["Target_Player"] == selected_player_name]
        if not player_rows.empty:
            st.dataframe(player_rows.sort_values("Buy_Probability(%)", ascending=False).reset_index(drop=True), use_container_width=True)
            # replot scatter with highlighted selected player marker
            fig_highlight = px.scatter(df_filtered, x="Expected_Bid", y="Buy_Probability(%)", color="Team",
                                       hover_data=["Target_Player", "Player_Role"], color_discrete_map=team_colors,
                                       title="Expected Bid vs Buy Probability (selected highlighted)")
            fig_highlight.add_scatter(x=player_rows["Expected_Bid"], y=player_rows["Buy_Probability(%)"],
                                      mode="markers+text", marker=dict(size=14, color="black", symbol="diamond"),
                                      text=player_rows["Target_Player"], textposition="top center", name="Selected Player")
            st.plotly_chart(fig_highlight, use_container_width=True)
        else:
            st.info("Selected player not found in full dataset.")
    else:
        st.subheader("Filtered / Selected Data (preview)")
        st.write(f"Rows matching filters: {len(df_filtered)}")
        st.dataframe(df_filtered.head(100), use_container_width=True)

# Download filtered data
@st.cache_data
def convert_df_to_csv(input_df):
    return input_df.to_csv(index=False).encode('utf-8')

csv = convert_df_to_csv(df_filtered)
st.download_button("Download filtered data as CSV", data=csv, file_name="ipl_filtered_data.csv", mime="text/csv")

# Small animation: animate the main bar quickly if user wants
if st.button("Quick Animate Bars"):
    anim_speed_local = max(0.02, 0.12 / anim_speed)
    agg_vals = agg.copy()
    teams_order = agg_vals["Team"].tolist()
    final_vals = agg_vals["Players_Needed"].tolist()
    steps = 20
    for step in range(1, steps + 1):
        intermediate = [int(v * step / steps) for v in final_vals]
        df_anim = pd.DataFrame({"Team": teams_order, "Players_Needed": intermediate})
        fig = px.bar(df_anim, x="Team", y="Players_Needed", text="Players_Needed", title="Animating Players Needed by Team",
                     color="Team", color_discrete_map=team_colors)
        fig.update_traces(marker_line_color='black', marker_line_width=1)
        st.plotly_chart(fig, use_container_width=True)
        time.sleep(anim_speed_local)
    # replaced balloons with IPL animation + sound
    play_ipl_theme_with_animation("Quick animation finished — enjoy the IPL vibe!")
    st.success("Quick animation finished.")
