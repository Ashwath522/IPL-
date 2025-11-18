# app.py — updated: clearer Players Needed chart + bat+ball animation + snappy interactions
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from streamlit_plotly_events import plotly_events
import time
import streamlit.components.v1 as components

st.set_page_config(layout="wide", page_title="IPL Auction Visualizer (Interactive)", page_icon="🏏")
st.title("IPL Auction Visualizer — Interactive & Happy")
st.markdown("Click a team bar to focus on a team (or pick a team in the sidebar). Short cricket animation available — quick and cheerful!")

# ---------------------------
# short cricket bat+ball animation HTML snippet (quick, ~0.8s)
# ---------------------------
def cricket_bat_ball_html():
    return r"""
    <style>
    .wrap { position: relative; width:100%; height:80px; overflow:hidden; margin:6px 0; }
    .ball { font-size:44px; position:absolute; left:-15%; top:18px; animation: slide 0.8s ease-out 0s 1 forwards; }
    .bat  { font-size:46px; position:absolute; left:-30%; top:2px; animation: swing 0.8s ease-out 0s 1 forwards; }
    @keyframes slide {
      0% { left:-15%; transform: rotate(0deg); opacity:0; }
      25% { opacity:1; transform: rotate(90deg); }
      60% { left:55%; transform: rotate(360deg); }
      100% { left:120%; transform: rotate(720deg); opacity:1; }
    }
    @keyframes swing {
      0% { left:-30%; transform: rotate(-30deg); opacity:0; }
      30% { left:10%; transform: rotate(0deg); opacity:1; }
      55% { left:40%; transform: rotate(10deg); }
      100% { left:110%; transform: rotate(25deg); opacity:1; }
    }
    .msg { font-weight:700; margin-top:6px; font-size:15px; }
    </style>
    <div class="wrap">
      <div class="bat">🏏</div>
      <div class="ball">🏐</div>
    </div>
    <div class="msg">That’s a lovely shot! ✨ Enjoy the view — short animation complete.</div>
    """

# ---------------------------
# Sample dataset generator (same as before)
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
# Data loading (uploader / fallback)
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
    st.warning(f"Uploaded dataset missing columns: {missing}. Using sample dataset instead.")
    df = generate_sample_dataset(400)

# numeric casting
for col in ["Purse_Remaining", "Expected_Bid", "Buy_Probability(%)", "Players_Needed", "Base_Price"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# prefer Player_Name if available
display_player_col = "Player_Name" if "Player_Name" in df.columns else "Target_Player"

# ---------------------------
# Filters & controls
# ---------------------------
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

# ---------------------------
# Apply filters
# ---------------------------
df_filtered = df.copy()
if teams_multiselect:
    df_filtered = df_filtered[df_filtered["Team"].isin(teams_multiselect)]
if "All" not in role_filter:
    df_filtered = df_filtered[df_filtered["Player_Role"].isin(role_filter)]
df_filtered = df_filtered[(df_filtered["Expected_Bid"] >= exp_range[0]) & (df_filtered["Expected_Bid"] <= exp_range[1])]
df_filtered = df_filtered[(df_filtered["Buy_Probability(%)"] >= prob_threshold)]

# ---------------------------
# Colors
# ---------------------------
palette = px.colors.qualitative.Plotly + px.colors.qualitative.Dark24 + px.colors.qualitative.Bold
unique_teams = sorted(df["Team"].dropna().unique())
team_colors = {team: palette[i % len(palette)] for i, team in enumerate(unique_teams)}

# ---------------------------
# Layout
# ---------------------------
col_main, col_side = st.columns([2, 1])

# ---- 1) MAIN grouped bar: Players Needed (sum) and Unique Targets (count)
with col_main:
    st.subheader("Team demand — Players Needed (sum) & Unique Target Count")
    # use filtered data so chart responds to filters
    agg_filtered = df_filtered.groupby("Team").agg(
        Total_Players_Needed=pd.NamedAgg(column="Players_Needed", aggfunc="sum"),
        Unique_Targets=pd.NamedAgg(column=display_player_col, aggfunc=lambda x: x.nunique())
    ).reset_index().sort_values("Total_Players_Needed", ascending=False)

    if agg_filtered.empty:
        st.info("No data to display for current filters.")
    else:
        # grouped bar chart so both metrics are visible
        fig_group = px.bar(
            agg_filtered.melt(id_vars="Team", value_vars=["Total_Players_Needed", "Unique_Targets"],
                              var_name="Metric", value_name="Value"),
            x="Team", y="Value", color="Metric", barmode="group",
            title="Players Needed (sum) vs Unique Targets (count) — filtered",
            text="Value",
            height=420
        )
        # apply team colors for the 'Team' x-axis ticks via consistent palette for other charts (bars are by Metric)
        fig_group.update_traces(texttemplate="%{text}", textposition="outside")
        st.plotly_chart(fig_group, use_container_width=True)

        # show numeric table under the chart for exact numbers (makes viewer happy — neat table)
        st.markdown("**Team demand table**")
        st.dataframe(agg_filtered.style.format({"Total_Players_Needed":"{:.0f}", "Unique_Targets":"{:.0f}"}), use_container_width=True)

    # capture clicks on the group chart by re-rendering a simple team-only bar (we still support plotly_events)
    # create a single-team bar chart variant for clickability
    # We'll present a compact single-series bar (Total_Players_Needed) for click-events
    if not agg_filtered.empty:
        fig_clickable = px.bar(agg_filtered, x="Team", y="Total_Players_Needed", text="Total_Players_Needed",
                               title="(Click team) — Total Players Needed", color="Team", color_discrete_map=team_colors, height=280)
        fig_clickable.update_traces(marker_line_color='black', marker_line_width=1)
        st.plotly_chart(fig_clickable, use_container_width=True)
        bar_click = plotly_events(fig_clickable, click_event=True, hover_event=False)
    else:
        bar_click = []

clicked_team = None
if bar_click:
    bp = bar_click[0]
    clicked_team = bp.get("x") or bp.get("label") or None

# if only one team selected in sidebar, prefer that
if len(teams_multiselect) == 1:
    clicked_team = teams_multiselect[0]

# ---- 2) Scatter: Expected Bid vs Buy Probability
with col_main:
    st.subheader("Expected Bid vs Buy Probability")
    if not df_filtered.empty:
        if clicked_team:
            color_map = {t: team_colors[t] if t == clicked_team else "lightgray" for t in unique_teams}
        else:
            color_map = team_colors
        fig_scatter = px.scatter(df_filtered, x="Expected_Bid", y="Buy_Probability(%)", color="Team",
                                 hover_data=[display_player_col, "Player_Role", "Base_Price"],
                                 title="Expected Bid vs Buy Probability", color_discrete_map=color_map, height=420)
        fig_scatter.update_layout(clickmode="event+select")
        st.plotly_chart(fig_scatter, use_container_width=True)
        scatter_click = plotly_events(fig_scatter, click_event=True, hover_event=False)
    else:
        st.info("No data for current filters.")
        scatter_click = []

# ---- 3) Box + Mean bar on side
with col_side:
    st.subheader("Expected Bid by Team")
    if not df_filtered.empty:
        fig_box = px.box(df_filtered, x="Team", y="Expected_Bid", color="Team", color_discrete_map=team_colors, points="outliers")
        st.plotly_chart(fig_box, use_container_width=True)
        mean_by_team = df_filtered.groupby("Team")["Expected_Bid"].mean().reset_index().sort_values("Expected_Bid", ascending=False)
        fig_mean = px.bar(mean_by_team, x="Team", y="Expected_Bid", text="Expected_Bid", color="Team", color_discrete_map=team_colors)
        fig_mean.update_traces(texttemplate="₹%{text:,.0f}", textposition="outside")
        st.plotly_chart(fig_mean, use_container_width=True)
    else:
        st.info("No data to display Expected Bid charts")

# ---- 4) Heatmap Team vs Role
with col_main:
    st.subheader("Team vs Role (heatmap)")
    if not df_filtered.empty:
        heat = df_filtered.groupby(["Team", "Player_Role"]).size().unstack(fill_value=0)
        fig_heat = px.imshow(heat, labels=dict(x="Player Role", y="Team", color="Count"), x=heat.columns, y=heat.index, aspect="auto", title="Players count by Team and Role")
        st.plotly_chart(fig_heat, use_container_width=True)
    else:
        st.info("No data for heatmap")

# ---- 5) Top players list
with col_side:
    st.subheader(f"Top {top_n_players} Players (by Buy Probability)")
    if not df_filtered.empty:
        top_players = (
            df_filtered.groupby(display_player_col)
            .agg({"Buy_Probability(%)": "max", "Expected_Bid": "max", "Team": lambda x: x.mode().iat[0] if not x.mode().empty else x.iloc[0]})
            .reset_index()
            .sort_values(["Buy_Probability(%)", "Expected_Bid"], ascending=[False, False])
            .head(int(top_n_players))
        )
        fig_top = px.bar(top_players, x=display_player_col, y="Buy_Probability(%)", color="Team",
                         color_discrete_map=team_colors, hover_data=["Expected_Bid"], title="Top Players — Buy Probability")
        fig_top.update_layout(xaxis_tickangle=-45, height=380)
        st.plotly_chart(fig_top, use_container_width=True)
    else:
        st.info("No data for top players")

# Optional sunburst
if enable_sunburst and not df_filtered.empty:
    st.subheader("Team → Role → Player (sunburst)")
    top_entries = (df_filtered.groupby(["Team", "Player_Role", display_player_col])
                   .agg({"Buy_Probability(%)": "max"})
                   .reset_index())
    top_entries = top_entries.sort_values("Buy_Probability(%)", ascending=False).head(200)
    fig_sun = px.sunburst(top_entries, path=["Team", "Player_Role", display_player_col], values="Buy_Probability(%)",
                          color="Team", color_discrete_map=team_colors, title="Sunburst: Team -> Role -> Player")
    st.plotly_chart(fig_sun, use_container_width=True)

# ---- Details & selection panel
st.markdown("---")
detail_col, preview_col = st.columns([1, 2])

# Determine selected player from scatter click
selected_player_name = None
if scatter_click:
    sp = scatter_click[0]
    cd = sp.get("customdata")
    hovertext = sp.get("hovertext")
    txt = sp.get("text")
    if isinstance(cd, (list, tuple)) and cd:
        for item in cd:
            if isinstance(item, str) and len(item) > 1:
                selected_player_name = item
                break
    if selected_player_name is None and isinstance(hovertext, str):
        selected_player_name = hovertext
    if selected_player_name is None and isinstance(txt, str):
        selected_player_name = txt
    # fallback coordinate
    if selected_player_name is None:
        x_val = sp.get("x")
        y_val = sp.get("y")
        if x_val is not None and y_val is not None:
            close = df_filtered[
                (np.isclose(df_filtered["Expected_Bid"], float(x_val), atol=1e-6)) &
                (np.isclose(df_filtered["Buy_Probability(%)"], float(y_val), atol=1e-6))
            ]
            if not close.empty:
                selected_player_name = close.iloc[0][display_player_col]

with detail_col:
    if clicked_team:
        st.subheader(f"Team: {clicked_team}")
        team_df = df[df["Team"] == clicked_team]
        if not team_df.empty:
            st.metric("Players in dataset", len(team_df))
            st.metric("Sum Players Needed", int(team_df["Players_Needed"].sum()))
            st.metric("Avg Expected Bid", f"₹{int(team_df['Expected_Bid'].mean()):,}")
            rcounts = team_df["Player_Role"].value_counts().reset_index()
            rcounts.columns = ["Player_Role", "Count"]
            st.table(rcounts)
            # handy quick animate team button (short)
            if st.button(f"Animate {clicked_team} (short)"):
                components.html(cricket_bat_ball_html(), height=120)
                st.success(f"{clicked_team} — nice animation! 😄")
        else:
            st.info("No records for this team.")
    else:
        st.subheader("Overview")
        st.markdown("Click a team bar or select a single team in the sidebar to view team details.")

with preview_col:
    if selected_player_name and enable_highlight:
        st.subheader(f"Selected Player: {selected_player_name}")
        player_rows = df[df[display_player_col] == selected_player_name]
        if not player_rows.empty:
            st.dataframe(player_rows.sort_values("Buy_Probability(%)", ascending=False).reset_index(drop=True), use_container_width=True)
            fig_highlight = px.scatter(df_filtered, x="Expected_Bid", y="Buy_Probability(%)", color="Team",
                                       hover_data=[display_player_col, "Player_Role"], color_discrete_map=team_colors,
                                       title="Expected Bid vs Buy Probability (selected highlighted)")
            fig_highlight.add_scatter(x=player_rows["Expected_Bid"], y=player_rows["Buy_Probability(%)"],
                                      mode="markers+text", marker=dict(size=14, color="black", symbol="diamond"),
                                      text=player_rows[display_player_col], textposition="top center", name="Selected Player")
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

# ---- Quick Animate Bars (short + happy)
if st.button("Quick Animate Bars (short)"):
    # quick growth animation by replotting a few steps only
    anim_speed_local = max(0.01, 0.08 / max(1, anim_speed))  # shorter and snappier
    if not agg_filtered.empty:
        teams_order = agg_filtered["Team"].tolist()
        final_vals = agg_filtered["Total_Players_Needed"].tolist()
        steps = 10  # short
        for step in range(1, steps + 1):
            intermediate = [int(v * step / steps) for v in final_vals]
            df_anim = pd.DataFrame({"Team": teams_order, "Players_Needed": intermediate})
            fig = px.bar(df_anim, x="Team", y="Players_Needed", text="Players_Needed",
                         color="Team", color_discrete_map=team_colors, title="Animating Players Needed (short)")
            fig.update_traces(marker_line_color='black', marker_line_width=1)
            st.plotly_chart(fig, use_container_width=True)
            time.sleep(anim_speed_local)
    # show short bat+ball animation and a cheerful message
    components.html(cricket_bat_ball_html(), height=120)
    st.success("That looked good! 😄 Enjoy the updated view.")

