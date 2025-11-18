# app.py — Interactive, shorter animations + IPL sound + cleaner money axes (lakhs)
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from streamlit_plotly_events import plotly_events
import streamlit.components.v1 as components
import time

st.set_page_config(layout="wide", page_title="IPL Auction Visualizer (Interactive)", page_icon="🏏")
st.title("IPL Auction Visualizer — Interactive & Celebratory")
st.markdown("Click a team bar to focus. Use animation buttons to run a short celebration (visual + IPL sound). Money shown in **lakhs (₹ ×100k)** for readability.")

# -------------------------
# small helpers
# -------------------------
@st.cache_data
def generate_sample_dataset(n_rows=200, seed=42):
    rng = np.random.RandomState(seed)
    teams = ["CSK", "MI", "RCB", "KKR", "SRH", "RR", "DC", "PBKS"]
    roles = ["Batsman", "Bowler", "All-Rounder", "Wicket-Keeper"]
    players = [f"Player_{i}" for i in range(1, 1001)]
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

def play_ipl_sound_and_show_animation(message="That’s a lovely shot!"):
    """Show a short bat+ball animation + play IPL 'ppp pppp' sound (WebAudio) and confetti text."""
    # JS uses WebAudio to play a short percussive 'ppp pppp' pattern and shows CSS animation
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
    // tiny WebAudio routine to play 2 quick 'ppp pppp' percussion sounds
    try {
      const ctx = new (window.AudioContext || window.webkitAudioContext)();
      const now = ctx.currentTime;
      // function to play short noise-like 'ppp'
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
      // schedule pattern: two groups of short blasts: p p p  (two groups)
      let t = now + 0.05;
      const short = 0.06;
      // first group (3 quick hits)
      playBlast(t, short, 0.04, 600); t += 0.09;
      playBlast(t, short, 0.04, 700); t += 0.09;
      playBlast(t, short, 0.04, 800); t += 0.2;
      // second group (3 quick hits)
      playBlast(t, short, 0.045, 600); t += 0.09;
      playBlast(t, short, 0.045, 700); t += 0.09;
      playBlast(t, short, 0.045, 800);
    } catch(e) {
      // ignore if WebAudio not available
      console.log("Audio error:", e);
    }
    </script>
    """
    components.html(js_html, height=170)

# -------------------------
# Data loading
# -------------------------
st.sidebar.header("Data Input")
uploaded_file = st.sidebar.file_uploader("Upload IPL dataset (Excel or CSV). Leave empty to use sample dataset.", type=["xlsx","csv"])
if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith(".xlsx"):
            df = pd.read_excel(uploaded_file)
        else:
            df = pd.read_csv(uploaded_file)
        st.sidebar.success("Loaded dataset: " + uploaded_file.name)
    except Exception as e:
        st.sidebar.error("Could not read file. Falling back to sample dataset. " + str(e))
        df = generate_sample_dataset(400)
else:
    df = generate_sample_dataset(400)
    st.sidebar.info("Using sample dataset (generated).")

# ensure required shape and types
required_cols = ["Team","Player_Role","Purse_Remaining","Target_Player","Expected_Bid","Buy_Probability(%)","Players_Needed","Base_Price"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    st.warning(f"Uploaded dataset is missing columns {missing}. Using generated sample instead.")
    df = generate_sample_dataset(400)

# numeric cast
for c in ["Purse_Remaining","Expected_Bid","Buy_Probability(%)","Players_Needed","Base_Price"]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

# prefer player's display column
display_player_col = "Player_Name" if "Player_Name" in df.columns else "Target_Player"

# Add human-friendly money axis columns (lakhs) for readability
df["Expected_Bid_L"] = (df["Expected_Bid"] / 1_00_000).round(2)
df["Base_Price_L"] = (df["Base_Price"] / 1_00_000).round(2)
df["Purse_Remaining_L"] = (df["Purse_Remaining"] / 1_00_000).round(2)

# -------------------------
# Filters & controls
# -------------------------
st.sidebar.header("Filters & Interactivity")
teams_list = sorted(df["Team"].dropna().unique())
teams_multiselect = st.sidebar.multiselect("Select teams", options=teams_list, default=teams_list)
role_options = sorted(df["Player_Role"].dropna().unique())
role_filter = st.sidebar.multiselect("Filter by role", options=["All"]+role_options, default=["All"])
exp_min = int(df["Expected_Bid_L"].min()) if not df["Expected_Bid_L"].isnull().all() else 0
exp_max = int(df["Expected_Bid_L"].max()) if not df["Expected_Bid_L"].isnull().all() else 2000
exp_range = st.sidebar.slider("Expected Bid range (lakhs ₹)", exp_min, exp_max, (exp_min, exp_max))
prob_threshold = st.sidebar.slider("Min Buy Probability (%)", 0, 100, 20)
anim_speed = st.sidebar.slider("Animation speed (lower = faster)", 1, 10, 4)
top_n_players = st.sidebar.number_input("Top N players", min_value=5, max_value=30, value=10)
st.sidebar.markdown("---")
enable_sunburst = st.sidebar.checkbox("Show Team→Role→Player sunburst", value=False)
enable_highlight = st.sidebar.checkbox("Highlight selected player", value=True)

# choose main metric
main_metric = st.sidebar.selectbox("Main chart metric", ["Total Players Needed", "Unique Targets", "Avg Expected Bid (L)"])

# Apply filters
df_filtered = df.copy()
if teams_multiselect:
    df_filtered = df_filtered[df_filtered["Team"].isin(teams_multiselect)]
if "All" not in role_filter:
    df_filtered = df_filtered[df_filtered["Player_Role"].isin(role_filter)]
df_filtered = df_filtered[(df_filtered["Expected_Bid_L"] >= exp_range[0]) & (df_filtered["Expected_Bid_L"] <= exp_range[1])]
df_filtered = df_filtered[df_filtered["Buy_Probability(%)"] >= prob_threshold]

# colors
palette = px.colors.qualitative.Plotly + px.colors.qualitative.Dark24 + px.colors.qualitative.Bold
unique_teams = sorted(df["Team"].dropna().unique())
team_colors = {team: palette[i % len(palette)] for i, team in enumerate(unique_teams)}

# -------------------------
# Aggregations for main display
# -------------------------
agg_filtered = None
if not df_filtered.empty:
    agg_filtered = df_filtered.groupby("Team").agg(
        Total_Players_Needed=pd.NamedAgg("Players_Needed","sum"),
        Unique_Targets=pd.NamedAgg(display_player_col, lambda x: x.nunique()),
        Avg_Expected_Bid_L=pd.NamedAgg("Expected_Bid_L", "mean")
    ).reset_index()

# Layout placeholders (for in-place updates)
col_main, col_side = st.columns([2,1])
group_ph = col_main.empty()
click_ph = col_main.empty()
scatter_ph = col_main.empty()
box_ph = col_side.empty()
mean_ph = col_side.empty()
top_ph = col_side.empty()
sun_ph = col_main.empty()

# -------------------------
# Main grouped chart (only if data)
# -------------------------
with col_main:
    st.subheader("Team demand — choose metric on the left")
    if agg_filtered is None or agg_filtered.empty:
        group_ph.info("No teams to display with current filters. Adjust filters or upload a dataset.")
        click_ph.empty()
    else:
        # build group chart depending on main_metric selection
        if main_metric == "Total Players Needed":
            plot_df = agg_filtered[["Team","Total_Players_Needed","Unique_Targets"]].copy()
            melted = plot_df.melt(id_vars="Team", value_vars=["Total_Players_Needed","Unique_Targets"], var_name="Metric", value_name="Value")
            fig_group = px.bar(melted, x="Team", y="Value", color="Metric", barmode="group", text="Value",
                               title="Players Needed (sum) vs Unique Targets (count) — filtered")
        else:
            if main_metric == "Unique Targets":
                fig_group = px.bar(agg_filtered, x="Team", y="Unique_Targets", color="Team", color_discrete_map=team_colors,
                                   title="Unique Target Players per Team", text="Unique_Targets")
            else: # Avg Expected Bid (L)
                fig_group = px.bar(agg_filtered, x="Team", y="Avg_Expected_Bid_L", color="Team", color_discrete_map=team_colors,
                                   title="Avg Expected Bid (in lakhs ₹)", text="Avg_Expected_Bid_L")
                fig_group.update_yaxes(title="Avg Expected Bid (₹ × 1e5 / lakhs)")

        fig_group.update_traces(textposition="outside")
        group_ph.plotly_chart(fig_group, use_container_width=True)

        # clickable single-series chart for selecting team (Total_Players_Needed)
        fig_click = px.bar(agg_filtered, x="Team", y="Total_Players_Needed", text="Total_Players_Needed", color="Team",
                           color_discrete_map=team_colors, title="(Click team) — Total Players Needed")
        fig_click.update_traces(marker_line_color='black', marker_line_width=1)
        click_ph.plotly_chart(fig_click, use_container_width=True)
        bar_click = plotly_events(fig_click, click_event=True, hover_event=False)

clicked_team = None
if 'bar_click' in locals() and bar_click:
    bp = bar_click[0]
    clicked_team = bp.get("x") or bp.get("label") or None
if len(teams_multiselect) == 1:
    clicked_team = teams_multiselect[0]

# -------------------------
# Scatter (use lakhs) and interaction
# -------------------------
with col_main:
    st.subheader("Expected Bid vs Buy Probability (lakhs)")
    if df_filtered.empty:
        scatter_ph.info("No points for current filters.")
        scatter_click = []
    else:
        if clicked_team:
            color_map = {t: team_colors[t] if t == clicked_team else "lightgray" for t in unique_teams}
        else:
            color_map = team_colors
        fig_scatter = px.scatter(df_filtered, x="Expected_Bid_L", y="Buy_Probability(%)", color="Team",
                                 hover_data=[display_player_col, "Player_Role", "Base_Price_L"],
                                 title="Expected Bid (lakhs) vs Buy Probability (%)", color_discrete_map=color_map, height=430)
        fig_scatter.update_layout(clickmode="event+select")
        fig_scatter.update_xaxes(title="Expected Bid (₹ ×100k = lakhs)")
        scatter_ph.plotly_chart(fig_scatter, use_container_width=True)
        scatter_click = plotly_events(fig_scatter, click_event=True, hover_event=False)

# -------------------------
# Side charts (box + mean + top players)
# -------------------------
with col_side:
    st.subheader("Expected Bid by Team")
    if df_filtered.empty:
        box_ph.info("No data")
        mean_ph.empty()
        top_ph.empty()
    else:
        fig_box = px.box(df_filtered, x="Team", y="Expected_Bid_L", color="Team", color_discrete_map=team_colors, points="outliers", title="Expected Bid (lakhs) distribution")
        fig_box.update_yaxes(title="Expected Bid (lakhs)")
        box_ph.plotly_chart(fig_box, use_container_width=True)
        mean_by_team = df_filtered.groupby("Team")["Expected_Bid_L"].mean().reset_index().sort_values("Expected_Bid_L", ascending=False)
        fig_mean = px.bar(mean_by_team, x="Team", y="Expected_Bid_L", color="Team", color_discrete_map=team_colors, text="Expected_Bid_L", title="Mean Expected Bid (lakhs)")
        fig_mean.update_traces(texttemplate="₹%{text:.2f}L", textposition="outside")
        mean_ph.plotly_chart(fig_mean, use_container_width=True)

        st.subheader(f"Top {top_n_players} Players")
        top_players = (
            df_filtered.groupby(display_player_col)
            .agg({"Buy_Probability(%)":"max", "Expected_Bid_L":"max", "Team": lambda x: x.mode().iat[0] if not x.mode().empty else x.iloc[0]})
            .reset_index()
            .sort_values(["Buy_Probability(%)","Expected_Bid_L"], ascending=[False, False])
            .head(int(top_n_players))
        )
        fig_top = px.bar(top_players, x=display_player_col, y="Buy_Probability(%)", color="Team", color_discrete_map=team_colors, hover_data=["Expected_Bid_L"], title="Top Players — Buy Probability")
        fig_top.update_layout(xaxis_tickangle=-45, height=320)
        top_ph.plotly_chart(fig_top, use_container_width=True)

# optional sunburst
if enable_sunburst and not df_filtered.empty:
    st.subheader("Team → Role → Player (sunburst)")
    top_entries = (df_filtered.groupby(["Team","Player_Role",display_player_col])
                   .agg({"Buy_Probability(%)":"max"}).reset_index().sort_values("Buy_Probability(%)", ascending=False).head(200))
    fig_sun = px.sunburst(top_entries, path=["Team","Player_Role",display_player_col], values="Buy_Probability(%)", color="Team", color_discrete_map=team_colors, title="Sunburst")
    sun_ph.plotly_chart(fig_sun, use_container_width=True)

# -------------------------
# Details & selection panel
# -------------------------
st.markdown("---")
left_col, right_col = st.columns([1,2])

# derive selected player from scatter click
selected_player_name = None
if 'scatter_click' in locals() and scatter_click:
    sp = scatter_click[0]
    cd = sp.get("customdata")
    hovertext = sp.get("hovertext")
    txt = sp.get("text")
    if isinstance(cd, (list,tuple)) and cd:
        for item in cd:
            if isinstance(item, str) and len(item)>1:
                selected_player_name = item
                break
    if selected_player_name is None and isinstance(hovertext,str):
        selected_player_name = hovertext
    if selected_player_name is None and isinstance(txt,str):
        selected_player_name = txt
    if selected_player_name is None:
        x_val = sp.get("x"); y_val = sp.get("y")
        if x_val is not None and y_val is not None:
            close = df_filtered[(np.isclose(df_filtered["Expected_Bid_L"], float(x_val), atol=1e-6)) & (np.isclose(df_filtered["Buy_Probability(%)"], float(y_val), atol=1e-6))]
            if not close.empty:
                selected_player_name = close.iloc[0][display_player_col]

with left_col:
    if clicked_team:
        st.subheader(f"Team: {clicked_team}")
        team_df = df[df["Team"]==clicked_team]
        if team_df.empty:
            st.info("No records for this team.")
        else:
            st.metric("Players in dataset", len(team_df))
            st.metric("Sum Players Needed", int(team_df["Players_Needed"].sum()))
            st.metric("Avg Expected Bid (L)", f"₹{team_df['Expected_Bid_L'].mean():.2f}L")
            rc = team_df["Player_Role"].value_counts().reset_index(); rc.columns=["Role","Count"]
            st.table(rc)
            if st.button(f"Animate {clicked_team} (short)"):
                # animate role pie & histogram and finish with sound+animation
                role_counts = team_df["Player_Role"].value_counts().reset_index(); role_counts.columns=["Player_Role","Count"]
                # pie animate
                pie_ph = left_col.empty()
                hist_ph = left_col.empty()
                steps = 10
                sleep_t = max(0.01, 0.06/ max(1, anim_speed))
                roles = role_counts["Player_Role"].tolist(); vals=role_counts["Count"].tolist()
                for s in range(1, steps+1):
                    inter = [int(v*s/steps) for v in vals]
                    pie_ph.plotly_chart(px.pie(pd.DataFrame({"Player_Role":roles,"Count":inter}), names="Player_Role", values="Count", title=f"{clicked_team} role (anim)"), use_container_width=True)
                    time.sleep(sleep_t)
                # histogram animate
                hist_vals, edges = np.histogram(team_df["Expected_Bid_L"], bins=6)
                for s in range(1, steps+1):
                    inter = (hist_vals*s/steps).astype(int)
                    df_hist = pd.DataFrame({"edge": edges[:-1], "count": inter})
                    hist_ph.plotly_chart(px.bar(df_hist, x="edge", y="count", title=f"{clicked_team} Expected Bid (lakhs) (anim)"), use_container_width=True)
                    time.sleep(sleep_t)
                play_ipl_sound_and_show_animation(f"{clicked_team} — Nice pick! Enjoy 🎉")
    else:
        st.subheader("Overview")
        st.markdown("Click a team bar or select a single team to view details and animate it.")

with right_col:
    if selected_player_name and enable_highlight:
        st.subheader(f"Selected Player: {selected_player_name}")
        pr = df[df[display_player_col]==selected_player_name]
        if not pr.empty:
            st.dataframe(pr.sort_values("Buy_Probability(%)", ascending=False).reset_index(drop=True), use_container_width=True)
            # highlight in scatter: replot the scatter with diamond marker for this player
            fig_h = px.scatter(df_filtered, x="Expected_Bid_L", y="Buy_Probability(%)", color="Team", hover_data=[display_player_col, "Player_Role"], color_discrete_map=team_colors)
            fig_h.add_scatter(x=pr["Expected_Bid_L"], y=pr["Buy_Probability(%)"], mode="markers+text", marker=dict(size=14, color="black", symbol="diamond"), text=pr[display_player_col], textposition="top center", name="Selected")
            scatter_ph.plotly_chart(fig_h, use_container_width=True)
        else:
            st.info("Selected player not found.")
    else:
        st.subheader("Filtered / Selected Data (preview)")
        st.write(f"Rows matching filters: {len(df_filtered)}")
        st.dataframe(df_filtered.head(100), use_container_width=True)

# -------------------------
# Quick animate bars (only when clicked)
# -------------------------
if st.button("Quick Animate Bars (short)"):
    if agg_filtered is not None and not agg_filtered.empty:
        anim_speed_local = max(0.01, 0.06/ max(1, anim_speed))
        teams_order = agg_filtered["Team"].tolist()
        final_vals = agg_filtered["Total_Players_Needed"].tolist()
        steps = 10
        for step in range(1, steps+1):
            intermediate = [int(v*step/steps) for v in final_vals]
            temp = agg_filtered.copy()
            temp["Total_Players_Needed"] = intermediate
            melt = temp.melt(id_vars="Team", value_vars=["Total_Players_Needed","Unique_Targets"], var_name="Metric", value_name="Value")
            group_ph.plotly_chart(px.bar(melt, x="Team", y="Value", color="Metric", barmode="group", text="Value", title="Animating players-needed (short)"), use_container_width=True)
            time.sleep(anim_speed_local)
    play_ipl_sound_and_show_animation("Short dashboard animation — lovely shot! 😄")

# -------------------------
# Download filtered
# -------------------------
@st.cache_data
def convert_df_to_csv(df_in):
    return df_in.to_csv(index=False).encode('utf-8')

csv = convert_df_to_csv(df_filtered)
st.download_button("Download filtered data as CSV", data=csv, file_name="ipl_filtered_data.csv", mime="text/csv")
