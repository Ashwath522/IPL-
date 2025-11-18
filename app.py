# app.py (animated edition)
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from streamlit_plotly_events import plotly_events
import time

st.set_page_config(layout="wide", page_title="IPL Auction Visualizer (Animated)", page_icon="🏏")

st.title("IPL Auction Visualizer — Interactive & Animated")
st.markdown(
    "Click a team bar or select a team. Use the **Animate** buttons to see animated transitions. "
    "Adjust speed for faster/slower animations."
)

@st.cache_data
def generate_sample_dataset(n_rows=200, seed=42):
    rng = np.random.RandomState(seed)
    teams = ["CSK", "MI", "RCB", "KKR", "SRH", "RR", "DC", "PBKS"]
    roles = ["Batsman", "Bowler", "All-Rounder", "Wicket-Keeper"]
    players = [f"Player_{i}" for i in range(1, 501)]

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
        df = generate_sample_dataset(200)
else:
    df = generate_sample_dataset(200)
    st.sidebar.info("Using sample dataset (200 rows).")

required_cols = ["Team", "Player_Role", "Purse_Remaining", "Target_Player", "Expected_Bid", "Buy_Probability(%)", "Players_Needed", "Base_Price"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    st.warning(f"Uploaded dataset is missing columns: {missing}. The app expects columns: {required_cols}. Using sample dataset instead.")
    df = generate_sample_dataset(200)

for col in ["Purse_Remaining", "Expected_Bid", "Buy_Probability(%)", "Players_Needed", "Base_Price"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Sidebar filters + animation speed
st.sidebar.header("Filters & Animation")
teams_list = sorted(df["Team"].unique())
team_select = st.sidebar.selectbox("Select team (also clickable in bar chart)", ["All"] + teams_list)
role_filter = st.sidebar.multiselect("Filter by role", options=["All"] + sorted(df["Player_Role"].unique()), default=["All"])
exp_min, exp_max = st.sidebar.slider("Expected Bid range (₹)", int(df["Expected_Bid"].min()), int(df["Expected_Bid"].max()), (int(df["Expected_Bid"].min()), int(df["Expected_Bid"].max())))
prob_threshold = st.sidebar.slider("Min Buy Probability (%)", 0, 100, 20)
anim_speed = st.sidebar.slider("Animation speed (lower = faster)", 1, 10, 4)

# Apply filters
df_filtered = df.copy()
if team_select != "All":
    df_filtered = df_filtered[df_filtered["Team"] == team_select]
if "All" not in role_filter:
    df_filtered = df_filtered[df_filtered["Player_Role"].isin(role_filter)]
df_filtered = df_filtered[(df_filtered["Expected_Bid"] >= exp_min) & (df_filtered["Expected_Bid"] <= exp_max)]
df_filtered = df_filtered[df_filtered["Buy_Probability(%)"] >= prob_threshold]

col1, col2 = st.columns([2, 1])

# placeholders for animated charts
bar_placeholder = col1.empty()
pie_placeholder = col1.empty()
scatter_placeholder = col1.empty()
team_pie_placeholder = col2.empty()
team_hist_placeholder = col2.empty()

# Bar chart (aggregate players needed per team)
agg = df.groupby("Team")["Players_Needed"].sum().reset_index().sort_values("Players_Needed", ascending=False)
fig_bar = px.bar(agg, x="Team", y="Players_Needed", text="Players_Needed", title="Total Players Needed by Team")
fig_bar.update_traces(marker_line_color='black', marker_line_width=1)
# show bar in placeholder
bar_placeholder.plotly_chart(fig_bar, use_container_width=True)
# capture clicks
selected_points = plotly_events(fig_bar, click_event=True, hover_event=False)

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
                             hover_data=["Target_Player", "Player_Role"], title="Expected Bid vs Buy Probability")
    scatter_placeholder.plotly_chart(fig_scatter, use_container_width=True)
else:
    scatter_placeholder.info("No points for current filters.")

with col2:
    clicked_team = None
    if selected_points:
        pt = selected_points[0]
        clicked_team = pt.get("x") or pt.get("label") or None
    if team_select != "All":
        clicked_team = team_select

    if clicked_team is None or clicked_team == "All":
        st.subheader("Overview / Strategy (All Teams)")
        st.markdown("Select a team (sidebar or click a bar) to view team-specific strategy. Use the animate button to make charts come alive.")
        top_teams = agg.head(8)
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
            if st.button("Animate Team"):
                # animate team pie growth + histogram
                animate_speed = max(0.02, 0.15 / anim_speed)  # scale speed slider to sleep time
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

# Global Animate button (bar + pie + scatter)
anim_col = st.container()
with anim_col:
    left, right = st.columns([1, 3])
    with left:
        if st.button("Animate Dashboard"):
            # animate bar growth
            anim_speed = max(0.02, 0.12 / anim_speed)  # lower -> faster
            # baseline: aggregate values
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
                time.sleep(anim_speed)
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
                    time.sleep(anim_speed)
            # quick scatter "pulse" effect (replot a few times)
            if not df_filtered.empty:
                for _ in range(3):
                    fig_s = px.scatter(df_filtered, x="Expected_Bid", y="Buy_Probability(%)", color="Team", title="Expected Bid vs Buy Probability (pulse)")
                    scatter_placeholder.plotly_chart(fig_s, use_container_width=True)
                    time.sleep(anim_speed / 2)
            st.balloons()
            st.success("Dashboard animation complete! 🎉")

# Bottom: filtered table + download
st.markdown("---")
st.subheader("Filtered / Selected Data (preview)")
st.write(f"Rows matching filters: {len(df_filtered)}")
st.dataframe(df_filtered.head(50), use_container_width=True)

@st.cache_data
def convert_df_to_csv(input_df):
    return input_df.to_csv(index=False).encode('utf-8')

csv = convert_df_to_csv(df_filtered)
st.download_button("Download filtered data as CSV", data=csv, file_name="ipl_filtered_data.csv", mime="text/csv")
