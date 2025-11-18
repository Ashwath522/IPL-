# app.py (interactive + extra charts edition)
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from streamlit_plotly_events import plotly_events
import time

st.set_page_config(layout="wide", page_title="IPL Auction Visualizer (Interactive)", page_icon="🏏")

st.title("IPL Auction Visualizer — Interactive & Enhanced")
st.markdown(
    "Click a team bar or select teams/players from the sidebar. New charts added: Top Target Players, Expected Bid by Team (box/mean), Team-Role heatmap and a Team->Role->Player sunburst. "
    "Use Animate buttons to see transitions. Click points in the scatter to select players."
)

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
        # create some repeated target players to produce meaningful top-player charts
        target = rng.choice(players, p=None)
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

# Sidebar: dataset upload or use sample
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

# Sidebar filters + animation speed + interactivity options
st.sidebar.header("Filters & Interactivity")
teams_list = sorted(df["Team"].dropna().unique())
teams_multiselect = st.sidebar.multiselect("Select teams (multi)", options=teams_list, default=teams_list)
role_options = sorted(df["Player_Role"].dropna().unique())
role_filter = st.sidebar.multiselect("Filter by role", options=["All"] + role_options, default=["All"])
exp_min = int(df["Expected_Bid"].min()) if not df["Expected_Bid"].isnull().all() else 0
exp_max = int(df["Expected_Bid"].max()) if not df["Expected_Bid"].isnull().all() else 1_00_00_000
exp_range = st.sidebar.slider("Expected Bid range (₹)", exp_min, exp_max, (exp_min, exp_max))
prob_threshold = st.sidebar.slider("Min Buy Probability (%)", 0, 100, 20)
base_min = int(df["Base_Price"].min()) if not df["Base_Price"].isnull().all() else 0
base_max = int(df["Base_Price"].max()) if not df["Base_Price"].isnull().all() else 1_00_00_000
base_range = st.sidebar.slider("Base Price range (₹)", base_min, base_max, (base_min, base_max))
anim_speed = st.sidebar.slider("Animation speed (lower = faster)", 1, 10, 4)
top_n_players = st.sidebar.number_input("Top N players to show", min_value=5, max_value=50, value=12, step=1)

st.sidebar.markdown("---")
st.sidebar.markdown("Interactivity:")
enable_highlight = st.sidebar.checkbox("Highlight selected player in scatter/table", value=True)
enable_sunburst = st.sidebar.checkbox("Show Team->Role->Player sunburst", value=True)

# Apply filters
df_filtered = df.copy()
if teams_multiselect:
    df_filtered = df_filtered[df_filtered["Team"].isin(teams_multiselect)]
if "All" not in role_filter:
    df_filtered = df_filtered[df_filtered["Player_Role"].isin(role_filter)]
df_filtered = df_filtered[(df_filtered["Expected_Bid"] >= exp_range[0]) & (df_filtered["Expected_Bid"] <= exp_range[1])]
df_filtered = df_filtered[(df_filtered["Buy_Probability(%)"] >= prob_threshold)]
df_filtered = df_filtered[(df_filtered["Base_Price"] >= base_range[0]) & (df_filtered["Base_Price"] <= base_range[1])]

col1, col2 = st.columns([2, 1])

# placeholders for interactive charts
bar_placeholder = col1.empty()
pie_placeholder = col1.empty()
scatter_placeholder = col1.empty()
team_pie_placeholder = col2.empty()
team_hist_placeholder = col2.empty()

# Bar chart (aggregate players needed per team) - main clickable chart
agg = df.groupby("Team")["Players_Needed"].sum().reset_index().sort_values("Players_Needed", ascending=False)
fig_bar = px.bar(agg, x="Team", y="Players_Needed", text="Players_Needed", title="Total Players Needed by Team")
fig_bar.update_traces(marker_line_color='black', marker_line_width=1)
bar_placeholder.plotly_chart(fig_bar, use_container_width=True)

# capture clicks on the bar chart
selected_points = plotly_events(fig_bar, click_event=True, hover_event=True)

# Pie and scatter (filtered)
role_agg = df_filtered["Player_Role"].value_counts().reset_index()
role_agg.columns = ["Player_Role", "Count"]
if not role_agg.empty:
    fig_pie = px.pie(role_agg, names="Player_Role", values="Count", title="Role Distribution (filtered)")
    pie_placeholder.plotly_chart(fig_pie, use_container_width=True)
else:
    pie_placeholder.info("No data for role distribution with current filters.")

if not df_filtered.empty:
    fig_scatter = px.scatter(df_filtered, x="Expected_Bid", y="Buy_Probability(%)", color="Team",
                             hover_data=["Target_Player", "Player_Role", "Base_Price"], title="Expected Bid vs Buy Probability")
    fig_scatter.update_layout(clickmode="event+select")
    scatter_placeholder.plotly_chart(fig_scatter, use_container_width=True)
    # capture clicks on scatter for player selection
    selected_players = plotly_events(fig_scatter, click_event=True, hover_event=False)
else:
    scatter_placeholder.info("No points for current filters.")
    selected_players = []

# Determine clicked_team from bar click OR if user selected a single team in the multiselect, prefer that
clicked_team = None
if selected_points:
    pt = selected_points[0]
    # depending on px.bar, "x" key holds team label
    clicked_team = pt.get("x") or pt.get("label") or None

if len(teams_multiselect) == 1:
    clicked_team = teams_multiselect[0]

# Show overview or team-specific panels
with col2:
    if clicked_team is None:
        st.subheader("Overview / Strategy (Selected Teams)")
        st.markdown("Select one team by clicking a bar or select a single team in the sidebar for team-specific details.")
        top_teams = agg[agg["Team"].isin(teams_multiselect)].head(8)
        st.table(top_teams.rename(columns={"Players_Needed": "Total Players Needed"}))
    else:
        st.subheader(f"Strategy — {clicked_team}")
        team_df = df[df["Team"] == clicked_team].sort_values(by=["Buy_Probability(%)", "Expected_Bid"], ascending=[False, True])
        if team_df.empty:
            st.info("No records for this team in the dataset.")
        else:
            st.metric("Total Purse (mean)", f"₹{int(team_df['Purse_Remaining'].mean()):,}")
            st.metric("Players in dataset", len(team_df))
            st.metric("Sum Players Needed", int(team_df["Players_Needed"].sum()))
            st.markdown("**Top target players (by Buy Probability)**")
            st.dataframe(team_df[["Target_Player", "Player_Role", "Expected_Bid", "Buy_Probability(%)", "Base_Price", "Players_Needed"]].head(12), use_container_width=True)

            st.markdown("**Team insights (click Animate Team to play)**")
            role_counts = team_df["Player_Role"].value_counts().reset_index()
            role_counts.columns = ["Player_Role", "Count"]
            if not role_counts.empty:
                fig_t_pie = px.pie(role_counts, names="Player_Role", values="Count", title=f"{clicked_team} - Role Breakdown (static)")
                team_pie_placeholder.plotly_chart(fig_t_pie, use_container_width=True)
            else:
                team_pie_placeholder.info("No role data for team.")

            fig_hist = px.histogram(team_df, x="Expected_Bid", nbins=8, title=f"{clicked_team} - Expected Bid Distribution (static)")
            team_hist_placeholder.plotly_chart(fig_hist, use_container_width=True)

            # Animate button for this selected team
            if st.button(f"Animate Team — {clicked_team}"):
                animate_speed = max(0.02, 0.15 / anim_speed)
                # Animate pie: grow slice counts from 0 -> true
                counts = role_counts.set_index("Player_Role")["Count"].to_dict()
                roles = list(counts.keys())
                final_vals = list(counts.values())
                steps = 20
                for s in range(1, steps + 1):
                    intermediate = [int(v * s / steps) for v in final_vals]
                    df_temp = pd.DataFrame({"Player_Role": roles, "Count": intermediate})
                    fig = px.pie(df_temp, names="Player_Role", values="Count", title=f"{clicked_team} - Role Breakdown (animating)")
                    team_pie_placeholder.plotly_chart(fig, use_container_width=True)
                    time.sleep(animate_speed)
                # Animate histogram bars growing
                hist_vals, edges = np.histogram(team_df["Expected_Bid"], bins=8)
                steps = 20
                for s in range(1, steps + 1):
                    intermediate = (hist_vals * s / steps).astype(int)
                    df_hist = pd.DataFrame({"bin": range(len(intermediate)), "count": intermediate, "edge": edges[:-1]})
                    fig_h = px.bar(df_hist, x="edge", y="count", labels={"edge": "Expected_Bid (₹)", "count": "Count"}, title=f"{clicked_team} - Expected Bid Distribution (animating)")
                    team_hist_placeholder.plotly_chart(fig_h, use_container_width=True)
                    time.sleep(animate_speed)
                st.balloons()
                st.success(f"Animation finished for {clicked_team} 🎉")

# Extra charts section (new)
st.markdown("## Additional Charts: Players & Expected Bids by Team")
extra_col1, extra_col2 = st.columns([2, 1])

with extra_col1:
    # Top target players by Buy Probability and Expected Bid
    if not df_filtered.empty:
        top_players = (
            df_filtered.groupby("Target_Player")
            .agg({"Buy_Probability(%)": "max", "Expected_Bid": "max", "Team": lambda x: x.mode().iat[0] if not x.mode().empty else x.iloc[0]})
            .reset_index()
            .sort_values(["Buy_Probability(%)", "Expected_Bid"], ascending=[False, False])
            .head(top_n_players)
        )
        st.markdown(f"### Top {top_n_players} Target Players (by Buy Probability then Expected Bid)")
        fig_top_players = px.bar(top_players, x="Target_Player", y="Buy_Probability(%)", color="Team",
                                 hover_data=["Expected_Bid"], title="Top Target Players — Buy Probability")
        fig_top_players.update_layout(xaxis_tickangle=-45, height=420)
        st.plotly_chart(fig_top_players, use_container_width=True)
    else:
        st.info("No data for top players with current filters.")

    # Expected Bid distribution by team: show box plot + mean bars
    if not df_filtered.empty:
        st.markdown("### Expected Bid by Team")
        fig_box = px.box(df_filtered, x="Team", y="Expected_Bid", points="all", title="Expected Bid distribution per Team (boxplot)")
        st.plotly_chart(fig_box, use_container_width=True)
        # show mean expected bid as a bar
        mean_by_team = df_filtered.groupby("Team")["Expected_Bid"].mean().reset_index().sort_values("Expected_Bid", ascending=False)
        fig_mean = px.bar(mean_by_team, x="Team", y="Expected_Bid", text="Expected_Bid", title="Mean Expected Bid per Team")
        fig_mean.update_traces(texttemplate="₹%{text:,.0f}", textposition="outside")
        st.plotly_chart(fig_mean, use_container_width=True)
    else:
        st.info("No data to show Expected Bid charts")

with extra_col2:
    # Heatmap of Team vs Role counts
    if not df_filtered.empty:
        st.markdown("### Team vs Role heatmap")
        heat = df_filtered.groupby(["Team", "Player_Role"]).size().unstack(fill_value=0)
        fig_heat = px.imshow(heat, labels=dict(x="Player Role", y="Team", color="Count"), x=heat.columns, y=heat.index, aspect="auto", title="Players count by Team and Role")
        st.plotly_chart(fig_heat, use_container_width=True)
    else:
        st.info("No data for heatmap")

    # Optional sunburst
    if enable_sunburst and not df_filtered.empty:
        st.markdown("### Team → Role → Top Player sunburst")
        # To keep sunburst reasonable, select top players per team-role
        top_entries = (df_filtered.groupby(["Team", "Player_Role", "Target_Player"])
                       .agg({"Buy_Probability(%)": "max", "Expected_Bid": "max"})
                       .reset_index())
        # pick top 100 by probability to avoid huge charts
        top_entries = top_entries.sort_values("Buy_Probability(%)", ascending=False).head(200)
        fig_sun = px.sunburst(top_entries, path=["Team", "Player_Role", "Target_Player"], values="Buy_Probability(%)", title="Sunburst: Team -> Role -> Player (by Buy Probability)")
        st.plotly_chart(fig_sun, use_container_width=True)

# Highlight selected player from scatter (if any)
selected_player_name = None
if selected_players:
    sp = selected_players[0]
    # Plotly scatter returns point with 'customdata' or 'hovertext' sometimes. We ensure hover_data included Target_Player so 'Target_Player' may be in point
    # The plotly_events returns 'customdata' or 'x' & 'y' — try to read 'points' keys
    selected_player_name = sp.get("customdata") or sp.get("hovertext") or sp.get("text") or None
    # If customdata provided as array-like, it may contain Target_Player as first element in our hover_data.
    if isinstance(selected_player_name, (list, tuple)) and len(selected_player_name) > 0:
        # try to find a string that looks like a player name
        for item in selected_player_name:
            if isinstance(item, str) and not item.startswith("Team:") and len(item) > 2:
                selected_player_name = item
                break
    # If still not a string, attempt to match by X/Y coords (Expected_Bid/Buy_Probability)
    if isinstance(selected_player_name, (int, float)) or selected_player_name is None:
        # fallback: use x and y
        x_val = sp.get("x")
        y_val = sp.get("y")
        if x_val is not None and y_val is not None:
            close = df_filtered[
                (np.isclose(df_filtered["Expected_Bid"], float(x_val), atol=1e-6)) &
                (np.isclose(df_filtered["Buy_Probability(%)"], float(y_val), atol=1e-6))
            ]
            if not close.empty:
                selected_player_name = close.iloc[0]["Target_Player"]

# Show selected player info and highlight (if enabled)
if selected_player_name and enable_highlight:
    st.markdown(f"### Selected Player: {selected_player_name}")
    player_rows = df[df["Target_Player"] == selected_player_name]
    if not player_rows.empty:
        st.dataframe(player_rows.sort_values("Buy_Probability(%)", ascending=False).reset_index(drop=True), use_container_width=True)
        # Replot scatter with highlight
        if not df_filtered.empty:
            fig_highlight = px.scatter(df_filtered, x="Expected_Bid", y="Buy_Probability(%)", color="Team",
                                       hover_data=["Target_Player", "Player_Role"], title="Expected Bid vs Buy Probability (selected highlighted)")
            fig_highlight.add_scatter(x=player_rows["Expected_Bid"], y=player_rows["Buy_Probability(%)"],
                                      mode="markers+text", marker=dict(size=14, color="red", symbol="diamond"),
                                      text=player_rows["Target_Player"], textposition="top center", name="Selected Player")
            st.plotly_chart(fig_highlight, use_container_width=True)
    else:
        st.info("Selected player not found in full dataset.")

# Global Animate button (bar + pie + scatter)
anim_col = st.container()
with anim_col:
    left, right = st.columns([1, 3])
    with left:
        if st.button("Animate Dashboard"):
            # animate bar growth
            anim_speed_local = max(0.02, 0.12 / anim_speed)
            agg_vals = agg.copy()
            teams_order = agg_vals["Team"].tolist()
            final_vals = agg_vals["Players_Needed"].tolist()
            steps = 25
            for step in range(1, steps + 1):
                intermediate = [int(v * step / steps) for v in final_vals]
                df_anim = pd.DataFrame({"Team": teams_order, "Players_Needed": intermediate})
                fig = px.bar(df_anim, x="Team", y="Players_Needed", text="Players_Needed", title="Total Players Needed by Team (animating)")
                fig.update_traces(marker_line_color='black', marker_line_width=1)
                bar_placeholder.plotly_chart(fig, use_container_width=True)
                time.sleep(anim_speed_local)
            # Animate the filtered role pie (if any values)
            if not role_agg.empty:
                roles = role_agg["Player_Role"].tolist()
                vals = role_agg["Count"].tolist()
                steps = 20
                for s in range(1, steps + 1):
                    inter = [int(v * s / steps) for v in vals]
                    df_temp = pd.DataFrame({"Player_Role": roles, "Count": inter})
                    figp = px.pie(df_temp, names="Player_Role", values="Count", title="Role Distribution (animating)")
                    pie_placeholder.plotly_chart(figp, use_container_width=True)
                    time.sleep(anim_speed_local)
            # quick scatter "pulse" effect (replot a few times)
            if not df_filtered.empty:
                for _ in range(3):
                    fig_s = px.scatter(df_filtered, x="Expected_Bid", y="Buy_Probability(%)", color="Team", title="Expected Bid vs Buy Probability (pulse)")
                    scatter_placeholder.plotly_chart(fig_s, use_container_width=True)
                    time.sleep(anim_speed_local / 2)
            st.balloons()
            st.success("Dashboard animation complete! 🎉")

# Bottom: filtered table + download
st.markdown("---")
st.subheader("Filtered / Selected Data (preview)")
st.write(f"Rows matching filters: {len(df_filtered)}")
st.dataframe(df_filtered.head(100), use_container_width=True)

@st.cache_data
def convert_df_to_csv(input_df):
    return input_df.to_csv(index=False).encode('utf-8')

csv = convert_df_to_csv(df_filtered)
st.download_button("Download filtered data as CSV", data=csv, file_name="ipl_filtered_data.csv", mime="text/csv")
