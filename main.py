import os
import json
import pandas as pd

import matplotlib.pyplot as plt
import fastf1
import streamlit as st
import scipy.stats
import anthropic
from dotenv import load_dotenv
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, fcluster

CLUSTER_PALETTE = ["blue", "orange", "green", "violet", "red", "gray"]

fastf1.set_log_level('WARNING')
fastf1.Cache.enable_cache('cache')  # Enable caching to speed up future runs

load_dotenv()
anthropic_client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

@st.cache_data
def get_races(year):
    schedule = fastf1.get_event_schedule(year)
    schedule = schedule[schedule['EventFormat'] != 'testing']
    return schedule['EventName'].tolist()

def get_session_data(year: int, grand_prix: str, session_type: str):
    session = fastf1.get_session(year, grand_prix, session_type)

    session.load()  # Load the session data

    laps = session.laps
    laps = laps[laps['LapNumber'] != 1] # first lap not valid because of starting from grid times
    laps = laps[laps['PitInTime'].isna()]
    laps = laps[laps['PitOutTime'].isna()]
    laps = laps[laps['TrackStatus'] == '1'] # Keep only fully green-flag laps (excludes any lap touched by yellow/SC/red/VSC)
    laps = laps[laps['Deleted'] == False]
    laps = laps[laps['LapTime'].notna()]

    # for each driver, filter out the lap times that are over 107% of their median pace
    driver_medians = laps.groupby('Driver')['LapTime'].transform('median')
    clean_laps = laps[laps['LapTime'] <= driver_medians * 1.07].copy()
    clean_laps = clean_laps[clean_laps['Compound'].isin(['SOFT', 'MEDIUM', 'HARD'])]  # Keep only laps on the main dry compounds to avoid outliers from wet/alternative tires
    clean_laps['LapTimeSeconds'] = clean_laps['LapTime'].dt.total_seconds()  # Convert LapTime to seconds for easier calculations

    tire_laps = laps[laps['Compound'].isin(['SOFT', 'MEDIUM', 'HARD'])].copy()
    tire_laps['LapTimeSeconds'] = tire_laps['LapTime'].dt.total_seconds()  
    # Optional: drop only the most extreme outliers (e.g., laps > 120% of median)
    # to catch real anomalies like off-track excursions
    driver_medians_tire = tire_laps.groupby('Driver')['LapTime'].transform('median')
    tire_laps = tire_laps[tire_laps['LapTime'] <= driver_medians_tire * 1.20]
    tire_profile = compute_tire_profile(tire_laps, min_stint_laps=10)
    
    features = clean_laps.groupby('Driver').agg(
        median_pace = ('LapTimeSeconds', 'median'),
        pace_std = ('LapTimeSeconds', 'std'),
        best_lap_delta=('LapTimeSeconds', lambda x: x.min() - x.median())
    )

    sector_medians = clean_laps.groupby('Driver').agg(
        s1_median = ('Sector1Time', 'median'),
        s2_median = ('Sector2Time', 'median'),
        s3_median = ('Sector3Time', 'median'),
    )

    sector_medians[['s1_median', 's2_median', 's3_median']] = sector_medians[['s1_median', 's2_median', 's3_median']].apply(lambda x: x.dt.total_seconds())

    # Fold per-compound degradation slopes into the feature matrix. Drivers who
    # didn't run a compound get the field median for that compound — they end up
    # neutral on that axis (z=0 post-scaling) so the cluster placement is driven
    # by the compounds they actually ran.
    if not tire_profile.empty and 'degradation_slope' in tire_profile.columns:
        tire_slopes = tire_profile['degradation_slope'].unstack('Compound')
        tire_slopes.columns = [f'deg_{c.lower()}' for c in tire_slopes.columns]
    else:
        tire_slopes = pd.DataFrame()
    for col in ['deg_soft', 'deg_medium', 'deg_hard']:
        if col not in tire_slopes.columns:
            tire_slopes[col] = pd.NA
    tire_slopes = tire_slopes[['deg_soft', 'deg_medium', 'deg_hard']]
    features = features.join(tire_slopes)
    tire_cols = ['deg_soft', 'deg_medium', 'deg_hard']
    features[tire_cols] = features[tire_cols].apply(lambda c: c.fillna(c.median()))
    # If a compound went entirely unused this race (e.g. wet conditions), drop
    # the column so it doesn't NaN out every driver via dropna below.
    features = features.dropna(axis=1, how='all')

    excluded_drivers = features[features.isna().any(axis=1)].index.tolist()
    features = features.dropna()

    if features.empty:
        return features, sector_medians.iloc[0:0], tire_profile, excluded_drivers

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    linked = linkage(scaled_features, method='ward')
    threshold = 0.7 * linked[:, 2].max()

    cluster_labels = fcluster(linked, t=threshold, criterion='distance')
    features['Cluster'] = cluster_labels
    sector_medians = sector_medians.join(features[['Cluster']], how='inner')

    return features, sector_medians, tire_profile, excluded_drivers

def fit_degradation(group: pd.DataFrame) -> pd.Series:
    """
    Run linear regression of lap time vs tire age for one (driver, compound) group.
    Returns slope (seconds gained per lap of tire life) and intercept (predicted
    lap time on fresh tires).
    """
    result = scipy.stats.linregress(group['TyreLife'], group['LapTimeSeconds'])
    return pd.Series({
        'degradation_slope': result.slope,
        'intercept': result.intercept,
        'r_squared': result.rvalue ** 2,
    })


def compute_tire_profile(clean_laps: pd.DataFrame, min_stint_laps: int) -> pd.DataFrame:
    """
    Build a per-(driver, compound) tire profile table.

    For each (driver, compound) combination, computes:
      - median_pace: median lap time on that compound
      - degradation_slope: seconds gained per lap of tire life (positive = degrading)
      - intercept: predicted lap time on fresh tires
      - r_squared: fit quality (0-1, higher = cleaner trend)
      - total_laps: number of clean laps on this compound
      - stint_count: number of distinct stints on this compound

    Filters out stints shorter than min_stint_laps (default 5) so noisy short
    stints don't pollute the regression.
    """
    # Restrict to dry-weather compounds — wet/intermediate break the degradation model
    dry_laps = clean_laps[clean_laps['Compound'].isin(['SOFT', 'MEDIUM', 'HARD'])]

    # Keep only stints with enough laps to fit a meaningful trend
    long_stints = dry_laps.groupby(['Driver', 'Stint']).filter(
        lambda g: len(g) >= min_stint_laps
    )

    if long_stints.empty:
        return pd.DataFrame()  # nothing survived the filter (extreme edge case)

    # Pass 1: regression slope + intercept per (driver, compound)
    degradation = long_stints.groupby(['Driver', 'Compound']).apply(
        fit_degradation, include_groups=False
    )

    # Pass 2: simple aggregations on the same grouping
    aggregates = long_stints.groupby(['Driver', 'Compound']).agg(
        median_pace=('LapTimeSeconds', 'median'),
        total_laps=('LapTimeSeconds', 'size'),
        stint_count=('Stint', 'nunique'),
    )

    # Join the two pieces on the (Driver, Compound) MultiIndex
    tire_profile = degradation.join(aggregates)

    return tire_profile

def compute_F_ratio(sector_medians):
    sector_medians = sector_medians.groupby('Cluster').filter(lambda x : len(x) > 1)  # Filter out clusters with only one member
    overall_mean = sector_medians[['s1_median', 's2_median', 's3_median']].mean()
    cluster_means = sector_medians.groupby('Cluster')[['s1_median', 's2_median', 's3_median']].mean()
    cluster_sizes = sector_medians['Cluster'].value_counts().sort_index()

    ss_between = ((cluster_means - overall_mean) ** 2).mul(cluster_sizes, axis=0).sum()    
    ss_within = sector_medians.groupby('Cluster')[['s1_median', 's2_median', 's3_median']].apply(lambda x: ((x - x.mean()) ** 2).sum()).sum()
    
    df_between = len(cluster_means) - 1
    df_within = len(sector_medians) - len(cluster_means)

    ms_between = ss_between / df_between
    ms_within = ss_within / df_within

    F_ratio = ms_between / ms_within
    return F_ratio, cluster_means[['s1_median', 's2_median', 's3_median']]


def describe_cluster(pace_tier: str, std_tier: str, delta_tier: str) -> str:
    """
    Map a cluster's feature tiers (low/med/high) to a human-readable label.

    Tiers are computed relative to the field of clusters for this race:
      - pace_tier: low = fastest, high = slowest
      - std_tier: low = most consistent, high = most variable
      - delta_tier: low = smallest best-to-median gap (least pace reserve),
                    high = largest gap (most pace reserve / most pushing)
    """
    # Front-runners: fast pace
    if pace_tier == 'low':
        if std_tier == 'low' and delta_tier == 'high':
            return "Controlled front-runners"
        elif std_tier in ('med', 'high') and delta_tier == 'low':
            return "Aggressive front-runners"

    # Midfield: medium pace
    if pace_tier == 'med':
        if std_tier == 'low' and delta_tier == 'low':
            return "Steady midfield"
        elif std_tier == 'high' and delta_tier == 'high':
            return "Eventful midfield"
        elif std_tier == 'high' and delta_tier == 'low':
            return "Aggressive midfield"

    # Back of the field: slow pace
    if pace_tier == 'high':
        if std_tier == 'low' and delta_tier == 'high':
            return "Steady backmarkers"
        elif std_tier == 'high' and delta_tier == 'low':
            return "Struggling / chaotic"

    # Fallback for any combination not explicitly handled
    return "Mixed race shape"


def build_cluster_summary(features: pd.DataFrame) -> dict:
    """
    Build a JSON-serializable summary of cluster characteristics for LLM labeling.

    Returns:
      {
        'field_means': {feature: value, ...},
        'clusters': {
          cluster_id: {'drivers': [...], 'size': int, 'means': {feature: value, ...}},
          ...
        }
      }
    """
    feature_cols = [c for c in features.columns if c != 'Cluster']
    field_means = {k: round(v, 3) for k, v in features[feature_cols].mean().items()}

    clusters = {}
    for cluster_id, members in features.groupby('Cluster'):
        clusters[int(cluster_id)] = {
            'drivers': members.index.tolist(),
            'size': int(len(members)),
            'means': {k: round(v, 3) for k, v in members[feature_cols].mean().items()},
        }

    return {'field_means': field_means, 'clusters': clusters}


@st.cache_data
def get_cluster_labels(summary: dict) -> dict:
    """
    Ask Claude to name + describe each cluster relative to the field. Falls back
    to the rule-based describe_cluster if the LLM call or JSON parse fails.

    Returns {cluster_id: {'name': str, 'description': str}}.
    """
    prompt = (
        "You are analyzing F1 driver clusters from a single Grand Prix race. Each "
        "cluster groups drivers with similar behavioral patterns.\n\n"
        "Features used for clustering:\n"
        "- median_pace: median lap time in seconds (lower = faster)\n"
        "- pace_std: standard deviation of lap times in seconds (lower = more consistent)\n"
        "- best_lap_delta: best lap minus median lap, in seconds (more negative = more pace held in reserve)\n"
        "- deg_soft / deg_medium / deg_hard: tire degradation slope in seconds per lap of tire life "
        "(positive = degrading; negative usually reflects fuel burn-off or track evolution rather than the tire actually improving)\n\n"
        f"Field-wide averages across all clusters:\n{json.dumps(summary['field_means'], indent=2)}\n\n"
        f"Clusters:\n{json.dumps(summary['clusters'], indent=2)}\n\n"
        "For each cluster, provide:\n"
        "- name: a short evocative label (3-5 words) capturing what makes this cluster stand out from the field\n"
        "- description: one sentence explaining the cluster's behavioral identity\n\n"
        "Phrase names RELATIVE to the field — what differentiates each cluster from the others. "
        "Avoid generic labels like 'Cluster 1' or 'Mixed group'.\n\n"
        "Respond with ONLY valid JSON, no markdown fences or commentary. Format:\n"
        '{"1": {"name": "...", "description": "..."}, "2": {"name": "...", "description": "..."}}'
    )

    try:
        response = anthropic_client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text.strip()
        # Claude often wraps JSON in ```json ... ``` fences despite instructions.
        # Slice between the first { and last } to tolerate fences or prose.
        if '{' in text and '}' in text:
            text = text[text.find('{'):text.rfind('}') + 1]
        parsed = json.loads(text)
        return {int(k): v for k, v in parsed.items()}
    except Exception as e:
        st.sidebar.warning(
            f"LLM labeling unavailable, using rule-based fallback ({type(e).__name__})."
        )
        cluster_means_df = pd.DataFrame(
            {cid: c['means'] for cid, c in summary['clusters'].items()}
        ).T
        tiers = pd.DataFrame({
            'pace': pd.qcut(cluster_means_df['median_pace'], q=3, labels=['low', 'med', 'high'], duplicates='drop'),
            'std': pd.qcut(cluster_means_df['pace_std'], q=3, labels=['low', 'med', 'high'], duplicates='drop'),
            'delta': pd.qcut(cluster_means_df['best_lap_delta'], q=3, labels=['low', 'med', 'high'], duplicates='drop'),
        })
        return {
            cid: {
                'name': describe_cluster(
                    tiers.loc[cid, 'pace'],
                    tiers.loc[cid, 'std'],
                    tiers.loc[cid, 'delta'],
                ),
                'description': '',
            }
            for cid in summary['clusters'].keys()
        }


st.title('F1 Driver Clustering')
st.write('This app clusters F1 drivers based on their performance in a given session')
with st.sidebar.form("race_selector"):
    year = st.selectbox("Season", [2022, 2023, 2024, 2025])
    races = get_races(year)
    race = st.selectbox("Race", races)
    submitted = st.form_submit_button("Load race")

if submitted:
    st.session_state.show_results = True
    st.session_state.selected_year = year
    st.session_state.selected_race = race

if st.session_state.get('show_results'):
    with st.spinner('Loading data and performing clustering...'):
        features, sector_medians, tire_profile, excluded_drivers = get_session_data(year, race, 'Race')

        if excluded_drivers:
            st.info(f"{len(excluded_drivers)} drivers excluded for insufficient data: {', '.join(excluded_drivers)}")

        if features.empty:
            st.warning(
                "No drivers had complete data across all three compounds. "
                "Try a race with more dry running, or lower min_stint_laps in compute_tire_profile."
            )
            st.stop()

        summary = build_cluster_summary(features)
        cluster_labels = get_cluster_labels(summary)

        cluster_ids = sorted(features['Cluster'].unique())
        st.caption(
            "Tire degradation bars show seconds gained per lap of tire life. "
            "Positive bars mean the tires are degrading; negative bars usually reflect fuel burn-off or "
            "track evolution rather than the tire actually improving."
        )
        cols = st.columns(len(cluster_ids))

        for cluster_id, col in zip(cluster_ids, cols):
            members = features[features['Cluster'] == cluster_id]
            field_mean = members[['median_pace', 'pace_std', 'best_lap_delta']].mean()
            with col:
                color = CLUSTER_PALETTE[(cluster_id - 1) % len(CLUSTER_PALETTE)]
                label_info = cluster_labels.get(cluster_id, {'name': '', 'description': ''})
                name = label_info.get('name', '')
                description = label_info.get('description', '')
                header = f"### :{color}[Cluster {cluster_id}] — {name}" if name else f"### :{color}[Cluster {cluster_id}]"
                st.markdown(header)
                if description:
                    st.markdown(f"*{description}*")
                st.write(f"{len(members)} drivers")
                st.write(", ".join(members.index.tolist()))
                st.metric(label="Median Pace", value=f"{field_mean['median_pace']:.2f}s", delta=f"{field_mean['median_pace'] - features['median_pace'].mean():.2f}s", delta_color="inverse")
                st.metric(label="Pace Std Dev", value=f"{field_mean['pace_std']:.2f}s", delta=f"{field_mean['pace_std'] - features['pace_std'].mean():.2f}s", delta_color="off")
                st.metric(label="Best Lap Delta", value=f"{field_mean['best_lap_delta']:.2f}s", delta=f"{field_mean['best_lap_delta'] - features['best_lap_delta'].mean():.2f}s", delta_color="off")

                tire_label = {'deg_soft': 'Soft', 'deg_medium': 'Medium', 'deg_hard': 'Hard'}
                tire_color = {'deg_soft': '#DA291C', 'deg_medium': '#F7C53F', 'deg_hard': '#DCDCDC'}
                available_tire_cols = [c for c in ['deg_soft', 'deg_medium', 'deg_hard'] if c in members.columns]
                if available_tire_cols:
                    tire_means = members[available_tire_cols].mean()
                    fig, ax = plt.subplots(figsize=(2.5, 1.3))
                    ax.barh(
                        [tire_label[c] for c in available_tire_cols],
                        [tire_means[c] for c in available_tire_cols],
                        color=[tire_color[c] for c in available_tire_cols],
                        edgecolor='#555',
                        linewidth=0.5,
                    )
                    ax.set_xlabel('Degradation (s/lap)', fontsize=7)
                    ax.tick_params(axis='both', labelsize=7)
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.invert_yaxis()
                    fig.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)
                
        with st.expander("Show driver-level data", expanded=False):
            st.dataframe(features.sort_values('Cluster'))

        st.subheader("Sector Discrimination")
        F_ratio, cluster_sector_means = compute_F_ratio(sector_medians)

        sector_name_map = {'s1_median': 'Sector 1', 's2_median': 'Sector 2', 's3_median': 'Sector 3'}
        F_ratio = F_ratio.rename(sector_name_map)
        cluster_sector_means = cluster_sector_means.rename(columns=sector_name_map)

        st.write(
            f"F-Ratio: S1: {F_ratio['Sector 1']:.2f} | "
            f"S2: {F_ratio['Sector 2']:.2f} | "
            f"S3: {F_ratio['Sector 3']:.2f}"
        )
        st.write("Cluster Means for Sector Times:")
        st.markdown(cluster_sector_means.style.format("{:.2f}").to_html(), unsafe_allow_html=True)

        winning_sector = F_ratio.idxmax()
        gap = cluster_sector_means[winning_sector].max() - cluster_sector_means[winning_sector].min()
        st.metric(
            label=f"Cluster gap in {winning_sector}",
            value=f"{gap:.2f}s",
            delta=f"F = {F_ratio.max():.1f}",
            delta_color="off"
        )

        st.subheader("Tire Profiles")
        st.dataframe(tire_profile.reset_index())