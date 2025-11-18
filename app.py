# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from streamlit_plotly_events import plotly_events

st.set_page_config(layout="wide", page_title="IPL Auction Visualizer", page_icon="🏏")

st.title("IPL Auction Visualizer — Interactive Dashboard")
st.markdown(
    "Click a bar (team) to filter the dashboard to that team, "
    "or choose a team from the dropdown. Upload your own dataset (Excel/CSV) or use the sample dataset."
)

#
# Utility: create a sample dataset if user doesn't upload
#
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
        purse = int(rng.randint(5, 20) * 1_00_00_000)  # in rupees
        target = rng.choice(players)
        expected = int(rng.randint(50, 200) * 1_00_000)  # expected bid
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

#
# Sidebar: dataset upload or use sample
#
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

# Basic sanity: required columns (if missing, try to map common names)
required_cols = ["Team", "Player_Role", "Purse_Remaining", "Target_Player", "Expected_Bid", "Buy_Probability(%)", "Players_Needed", "Base_Price"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    st.warning(f"Uploaded dataset is missing columns: {missing}. The app expects columns: {required_cols}. Using sample dataset instead.")
    df = generate_sample_dataset(200)

# Convert numeric types
for col in ["Purse_Remaining", "Expected_Bid", "Buy_Probability(%)", "Players_Needed", "Base_Price"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Sidebar: filters
st.sidebar.header("Filters")
teams_list = sorted(df["Team"].unique())
team_select = st.sidebar.selectbox("Select team (also clickable in bar chart)", ["All"] + teams_list)
role_filter = st.sidebar.multiselect("Filter by role", options=["All"] + sorted(df["Player_Role"].unique()), default=["All"])
exp_min, exp_max = st.sidebar.slider("Expected Bid range (₹)", int(df["Expected_Bid"].min()), int(df["Expected_Bid"].max()), (int(df["Expected_Bid"].min()), int(df["Expected_Bid"].max())))
prob_threshold = st.sidebar.slider("Min Buy Probability (%)", 0, 100, 20)

# Apply filters
df_filtered = df.copy()
if team_select != "All":
    df_filtered = df_filtered[df_filtered["Team"] == team_select]
if "All" not in role_filter:
    df_filtered = df_filtered[df_filtered["Player_Role"].isin(role_filter)]
df_filtered = df_filtered[(df_filtered["Expected_Bid"] >= exp_min) & (df_filtered["Expected_Bid"] <= exp_max)]
df_filtered = df_filtered[df_filtered["Buy_Probability(%)"] >= prob_threshold]

# Layout: left (charts) and right (details)
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Players Needed per Team (click a bar)")
    # aggregate players needed per team (sum or mean?)
    agg = df.groupby("Team")["Players_Needed"].sum().reset_index().sort_values("Players_Needed", ascending=False)
    fig_bar = px.bar(agg, x="Team", y="Players_Needed", text="Players_Needed", title="Total Players Needed by Team",
                     labels={"Players_Needed": "Players Needed (sum)"})
    fig_bar.update_traces(marker_line_color='black', marker_line_width=1)
    # use plotly_events to capture clicks
    selected_points = plotly_events(fig_bar, click_event=True, hover_event=False)
    st.plotly_chart(fig_bar, use_container_width=True)

    # Pie chart for roles (global or filtered by selected team later)
    st.subheader("Role distribution (filtered)")
    role_agg = df_filtered["Player_Role"].value_counts().reset_index()
    role_agg.columns = ["Player_Role", "Count"]
    if role_agg.empty:
        st.info("No data to show for current filters.")
    else:
        fig_pie = px.pie(role_agg, names="Player_Role", values="Count", title="Role Distribution")
        st.plotly_chart(fig_pie, use_container_width=True)

    # Scatter: Expected Bid vs Probability (filtered)
    st.subheader("Expected Bid vs Buy Probability (filtered)")
    if df_filtered.empty:
        st.info("No players match current filter criteria.")
    else:
        fig_scatter = px.scatter(
            df_filtered,
            x="Expected_Bid",
            y="Buy_Probability(%)",
            color="Team",
            hover_data=["Target_Player", "Player_Role", "Base_Price", "Players_Needed"],
            title="Expected Bid (₹) vs Buy Probability (%)"
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

with col2:
    # Determine which team was clicked (plotly_events returns list of dicts with 'x' etc)
    clicked_team = None
    if selected_points:
        # Each point has e.g. {'curveNumber': 0, 'pointNumber': 2, 'x': 'RCB', 'y': 34}
        # We'll take the 'x' value if present
        pt = selected_points[0]
        clicked_team = pt.get("x") or pt.get("label") or None

    # If user selected a team in sidebar, prefer that selection
    if team_select != "All":
        clicked_team = team_select

    if clicked_team is None or clicked_team == "All":
        st.subheader("Overview / Strategy (All Teams)")
        st.markdown("Click on a team bar or select a team from the sidebar to view the team's strategy and top target players.")
        # Show top teams by players needed
        top_teams = agg.head(8)
        st.table(top_teams.rename(columns={"Players_Needed": "Total Players Needed"}))
    else:
        st.subheader(f"Strategy — {clicked_team}")
        team_df = df[df["Team"] == clicked_team].sort_values(by=["Buy_Probability(%)", "Expected_Bid"], ascending=[False, True])
        if team_df.empty:
            st.info("No records for this team in the dataset.")
        else:
            # Basic metrics
            st.metric("Total Purse (example value shown as mean of records)", f"₹{int(team_df['Purse_Remaining'].mean()):,}")
            st.metric("Players in dataset", len(team_df))
            st.metric("Sum Players Needed", int(team_df["Players_Needed"].sum()))

            st.markdown("**Top target players (by Buy Probability)**")
            st.dataframe(team_df[["Target_Player", "Player_Role", "Expected_Bid", "Buy_Probability(%)", "Base_Price", "Players_Needed"]].head(12), use_container_width=True)

            # small charts for the selected team
            st.markdown("**Team insights**")
            col_a, col_b = st.columns(2)
            with col_a:
                role_counts = team_df["Player_Role"].value_counts().reset_index()
                role_counts.columns = ["Player_Role", "Count"]
                if not role_counts.empty:
                    fig_t_pie = px.pie(role_counts, names="Player_Role", values="Count", title=f"{clicked_team} - Role Breakdown")
                    st.plotly_chart(fig_t_pie, use_container_width=True)
            with col_b:
                # expected bid distribution
                fig_hist = px.histogram(team_df, x="Expected_Bid", nbins=10, title=f"{clicked_team} - Expected Bid Distribution (₹)")
                st.plotly_chart(fig_hist, use_container_width=True)

# Bottom: show raw filtered table and export options
st.markdown("---")
st.subheader("Filtered / Selected Data (preview)")
st.write(f"Rows matching filters: {len(df_filtered)}")
st.dataframe(df_filtered.head(50), use_container_width=True)

# Allow download of filtered data as CSV
@st.cache_data
def convert_df_to_csv(input_df):
    return input_df.to_csv(index=False).encode('utf-8')

csv = convert_df_to_csv(df_filtered)
st.download_button("Download filtered data as CSV", data=csv, file_name="ipl_filtered_data.csv", mime="text/csv")

st.caption("Tip: To deploy on Streamlit Cloud, push this repo containing `app.py` and `requirements.txt`. "
           "You can upload your actual auction dataset (Excel/CSV) from the sidebar to visualize real data.")
