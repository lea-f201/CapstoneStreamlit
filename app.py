import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.markdown("""
<style>
/* Section spacing */
.block-container { padding-top: 1.2rem; }

/* Card for list items */
.card {
  padding: 10px 12px; margin: 6px 0;
  border: 1px solid #e9edf3; border-radius: 12px;
  background: #fbfcfe;
}

/* Name + ID line */
.item-name { font-weight: 600; }
.item-id { color: #64748b; font-size: 12px; margin-left: 6px; }

/* Tiny secondary text */
.small { color:#94a3b8; font-size:12px; }

/* Make small buttons look neat */
.stButton>button {
  padding: 3px 10px; border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# --- Load Data ---
df = pd.read_csv("df_ensemble_all_predictions.csv")

# ==== Recommendation helpers & cached loaders ====

# --- Cached parquet loaders ---
@st.cache_data(show_spinner=False)
def load_parquets():
    items = pd.read_parquet("items.parquet")                     # product_id, product_name
    segp  = pd.read_parquet("segment_popularity.parquet")        # segment, product_id, popularity_score, rank
    sims  = pd.read_parquet("similarity_matrix.parquet")         # FULL cosine matrix (rows/cols = product_id)
    return items, segp, sims

@st.cache_data(show_spinner=False)
def load_segment_spend_summary():
    # Expecting columns: segment, average_spend, median_spend
    seg = pd.read_csv("segment_spend_summary.csv")
    seg.columns = [c.strip().lower() for c in seg.columns]
    # sanity
    required = {"segment", "average_spend", "median_spend"}
    missing = required - set(seg.columns)
    if missing:
        st.warning(f"segment_spend_summary.csv is missing columns: {missing}")
    return seg

@st.cache_data(show_spinner=False)
def load_rfm_segments_only():
    rfm = pd.read_csv("rfm_scores.csv", usecols=["Customer/ID", "Segment"])
    rfm["Customer/ID"] = rfm["Customer/ID"].astype(str)
    return rfm  # columns: Customer/ID, Segment
# --- Maps & utilities ---
def ids_to_names(items_df: pd.DataFrame, ids: list[str]) -> list[str]:
    lut = dict(zip(items_df["product_id"], items_df["product_name"]))
    return [lut.get(pid, pid) for pid in ids]

def names_to_ids(items_df: pd.DataFrame, names: list[str]) -> list[str]:
    lut = dict(zip(items_df["product_name"], items_df["product_id"]))
    return [lut[n] for n in names if n in lut]

def recommend_from_matrix(basket_ids: list[str], sim_df: pd.DataFrame, top_n: int) -> list[str]:
    """
    Sum similarity across all items in basket using dense cosine matrix.
    Drops basket items; returns list of product_ids (length up to top_n).
    """
    if not basket_ids or sim_df is None or sim_df.empty:
        return []
    valid = [p for p in basket_ids if p in sim_df.columns]
    if not valid:
        return []
    # sum columns corresponding to the basket
    scores = sim_df[valid].sum(axis=1)
    scores = scores.drop(labels=basket_ids, errors="ignore")
    # largest scores first
    return scores.nlargest(top_n).index.tolist()

def segment_top_products(segpop: pd.DataFrame, segment: str, k: int, exclude: set[str]) -> list[str]:
    if not segment:
        return []
    df = (segpop[segpop["segment"] == segment]
          .sort_values(["rank", "popularity_score"], ascending=[True, False]))
    res = [pid for pid in df["product_id"].tolist() if pid not in exclude]
    return res[:k]

def random_fill(items_df: pd.DataFrame, k: int, exclude: set[str]) -> list[str]:
    if k <= 0:
        return []
    pool = items_df.loc[~items_df["product_id"].isin(exclude), "product_id"].tolist()
    if not pool:
        return []
    rng = np.random.default_rng(123)
    take = min(k, len(pool))
    return list(rng.choice(pool, size=take, replace=False))


# --- Sidebar Layout ---
st.sidebar.title("Identify, Prioritize, Engage: A Data-Driven Framework  for Optimizing Customer Relationships ")
dashboard_choice = st.sidebar.selectbox(
    "Choose Dashboard View",
    ["Churn Cost Analysis", "Segmentation", "Collaborative Filtering"]
)


# ------------------ DASHBOARD 1: CHURN COST ANALYSIS ------------------
if dashboard_choice == "Churn Cost Analysis":
    st.title("Identify - Churn Cost Analysis Dashboard")

    # --- Create Columns: Left (1/3) = Inputs, Right (2/3) = Tables + Text ---
    col_inputs, col, col_graph = st.columns([1, 0.1, 2])

    with col_inputs:
        st.header("Business Cost Parameters")

        # Toggle: use per-customer CLV from segment table
        use_segment_clv = st.checkbox("Use segment-based CLV per customer", value=False)
        clv_basis = None
        if use_segment_clv:
            clv_basis = st.radio("CLV basis (from segment table)",
                                 ["average_spend", "median_spend"],
                                 horizontal=True, index=0)
            st.caption("Global CLV input is ignored when this is enabled.")
            clv = st.number_input("Global CLV (ignored when using segment-based CLV)",
                                  min_value=0.0, value=300.0, step=10.0, disabled=True)
        else:
            clv = st.number_input("Customer Lifetime Value (CLV) - 6 months",
                                  min_value=0.0, value=300.0, step=10.0)

        profit_margin = st.slider("Profit Margin (as decimal)", min_value=0.0, max_value=1.0, value=0.2)

        st.subheader("Marketing Costs")
        cost_employee_visit = st.number_input("Employee Visit Cost", min_value=0.0, value=15.0)
        cost_promo_discount = st.number_input("Promotion Discount", min_value=0.0, value=20.0)
        effectiveness_rate = st.slider("Promotion Effectiveness Rate", min_value=0.0, max_value=1.0, value=0.5)
        cost_sms = st.number_input("SMS Cost", min_value=0.0, value=0.05)
        cost_call = st.number_input("Call Cost", min_value=0.0, value=0.2)
        cost_other = st.number_input("Other Marketing Costs", min_value=0.0, value=0.0)

    # Persist business parameters for use in other tabs
    st.session_state["biz_params"] = {
        "clv": clv,
        "profit_margin": profit_margin,
        "cost_employee_visit": cost_employee_visit,
        "cost_promo_discount": cost_promo_discount,
        "effectiveness_rate": effectiveness_rate,
        "cost_sms": cost_sms,
        "cost_call": cost_call,
        "cost_other": cost_other,
    }

    # --- Step 2: Cost Calculation ---
    # Pretty names for models (used for display)
    pretty_names = {
        "logreg_proba": "Logistic Regression",
        "rf_proba": "Random Forest",
        "lgbm_final_proba": "LightGBM",
        "attn_proba": "LSTM + Attention",
        "hybrid_proba": "Hybrid LSTM + Static",
        "weighted_ensemble_proba": "Weighted Ensemble",
        "voted_pred_best": "Voting Classifier (Predictions)"  # not a proba column
    }

    # Probability columns in predictions file
    model_cols = [c for c in df.columns if c.endswith("_proba")]
    if "true_label" in model_cols:
        model_cols.remove("true_label")

    # Evaluation population size
    if "true_label" in df.columns:
        n_eval = int((~pd.isna(df["true_label"])).sum())
    else:
        n_eval = len(df)

    # Ensure ID is str and grab true labels if present
    df["Customer/ID"] = df["Customer/ID"].astype(str)
    true_labels = df["true_label"].values if "true_label" in df.columns else None

    # --- Build per-customer CLV vector (aligned to df rows) ---
    if use_segment_clv:
        seg_table = load_segment_spend_summary()  # columns: segment, average_spend, median_spend
        rfm_seg   = load_rfm_segments_only()      # columns: Customer/ID, Segment

        seg_table = seg_table.rename(columns={"segment": "Segment"})
        seg_basis = seg_table[["Segment", clv_basis]].copy()
        seg_basis.columns = ["Segment", "clv_segment_basis"]

        cust_to_seg = rfm_seg.merge(seg_basis, on="Segment", how="left")
        clv_lookup = dict(zip(cust_to_seg["Customer/ID"], cust_to_seg["clv_segment_basis"]))

        clv_series = df["Customer/ID"].map(clv_lookup)
        clv_series = pd.to_numeric(clv_series, errors="coerce").fillna(clv)
        clv_vec = clv_series.values
    else:
        clv_vec = np.full(len(df), clv, dtype=float)

    # --- Cost sweep across thresholds using per-customer CLV ---
    thresholds = np.linspace(0.05, 0.95, 19)
    cost_results = {}

    def total_cost_for(series, t, clv_vec):
        preds = (series >= t).astype(int)
        if true_labels is not None:
            fp_mask = (preds == 1) & (true_labels == 0)
            fn_mask = (preds == 0) & (true_labels == 1)

            fp = int(fp_mask.sum())
            cost_fp = fp * (
                cost_employee_visit + cost_sms + cost_call + cost_other +
                (cost_promo_discount * profit_margin * effectiveness_rate)
            )
            # Per-customer CLV for false negatives
            cost_fn = float(clv_vec[fn_mask].sum() * profit_margin)
            return cost_fp + cost_fn
        else:
            # No ground truth → outreach cost only
            fp = int(((series >= t).astype(int) == 1).sum())
            cost_fp = fp * (
                cost_employee_visit + cost_sms + cost_call + cost_other +
                (cost_promo_discount * profit_margin * effectiveness_rate)
            )
            return cost_fp

    for model in model_cols:
        s = df[model].values
        costs = []
        for thresh in thresholds:
            total_cost = total_cost_for(s, thresh, clv_vec)
            costs.append(total_cost)
        cost_results[model] = costs

    cost_df = pd.DataFrame(cost_results, index=thresholds)
    best_cost_per_model = cost_df.min()

    # ---------------- RIGHT: TABLES (no plots) ----------------
    with col_graph:
        # 1) PER-SEGMENT MEAN & MEDIAN CLV TABLE (unchanged)
        st.subheader("Segment CLV Table (Mean & Median)")
        st.caption("Calculated using a rolling 6-month window of customer spending, aggregated per segment, then averaged and medianed across months.")
        seg_tbl_show = load_segment_spend_summary().rename(columns={
            "segment": "Segment",
            "average_spend": "Average Spend",
            "median_spend": "Median Spend"
        })
        st.dataframe(
            seg_tbl_show.reset_index(drop=True).style.format({"Average Spend": "${:,.2f}", "Median Spend": "${:,.2f}"}),
            use_container_width=True
        )
        st.download_button(
            "📥 Download Segment CLV Table",
            data=seg_tbl_show.to_csv(index=False),
            file_name="segment_spend_summary.csv",
            mime="text/csv",
            use_container_width=True
        )

        # 2) Lowest cost per model — show TOP 3 only, normalized per 100 customers
        denom = max(n_eval, 1)  # avoid div-by-zero
        scale = 100.0 / denom
        best_cost_per100 = best_cost_per_model * scale  # per 100 customers
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader(f"Lowest Cost per Model (per 100 customers)")
        if use_segment_clv:
            st.caption(
                f"Costs use per-customer CLV from the segment table "
                f"({clv_basis.replace('_',' ')}), falling back to the global CLV where missing."
            )

        # Show ONLY the top 3
        top3_models = best_cost_per100.sort_values().index[:3]
        for model in top3_models:
            pretty = pretty_names.get(model, model)
            min_cost_per100 = best_cost_per100[model]
            best_thresh = cost_df[model].idxmin()
            st.markdown(f"✅ **{pretty}** → Cost: `${min_cost_per100:,.2f}` per 100 customers at Threshold: `{best_thresh:.2f}`")

        # 3) Collapsible table for ALL models (per 100 customers), closed by default
        with st.expander("check all models results", expanded=False):
            best_table = (
                pd.DataFrame({
                    "Model": [pretty_names.get(m, m) for m in best_cost_per100.index],
                    "Best Cost (per 100 customers)": best_cost_per100.values,
                    "Best Threshold": [cost_df[m].idxmin() for m in best_cost_per100.index],
                })
                .sort_values("Best Cost (per 100 customers)", ascending=True)
            )
            st.dataframe(
                best_table.style.format({"Best Cost (per 100 customers)": "${:,.2f}",
                                        "Best Threshold": "{:.2f}"}),
                use_container_width=True
            )


# ---------------------- DASHBOARD 2: SEGMENTATION ------------------
elif dashboard_choice == "Segmentation":
    st.title("Prioritize - Customer Segmentation Dashboard")

    # --- Load precomputed RFM data ---
    rfm = pd.read_csv("rfm_scores.csv")

    # --- Small style helpers ---
    st.markdown("""
    <style>
      .panel{background:#ffffff;border:1px solid #e9edf3;border-radius:14px;padding:14px 16px;margin-bottom:14px}
      .panel h3{margin:0 0 6px 0}
      .badge{background:#eef2ff;color:#3730a3;border-radius:999px;padding:2px 8px;font-size:12px;margin-left:6px}
      .kpi{background:#fbfcfe;border:1px solid #e9edf3;border-radius:12px;padding:10px 12px}
      .soft{color:#64748b}
    </style>
    """, unsafe_allow_html=True)

    # --- Segment positions for the RFM grid (same as before) ---
    segment_positions = {
        "Recent Users":          (4, 4, 2, 1),
        "Promising":             (4, 5, 1, 1),
        "About To Sleep":        (3, 4, 1, 2),
        "Hibernating":           (1, 2, 2, 2),
        "Lost":                  (1, 4, 2, 2),
        "Price Sensitive":       (5, 5, 1, 1),
        "Can't Lose Them":       (1, 1, 2, 1),
        "Need Attention":        (3, 3, 1, 1),
        "Loyal Customers":       (3, 1, 2, 2),
        "Champions":             (5, 1, 1, 2),
        "Potential Loyalists":   (4, 3, 2, 1)
    }

    # --- Summary numbers per segment ---
    summary = rfm['Segment'].value_counts().reset_index()
    summary.columns = ['Segment', 'Count']
    summary['Percent'] = (summary['Count'] / len(rfm) * 100).round(2)

    segment_counts   = dict(zip(summary['Segment'], summary['Count']))
    segment_percents = dict(zip(summary['Segment'], summary['Percent']))
    total_revenue    = rfm['monetary'].sum()

    # --- Layout: left = grid, right = info & exporter ---
    left,mid, right = st.columns([1.6,0.1, 1])

    # Sidebar segment picker (also mirrored on the right panel title)
    st.sidebar.subheader("Select Segment")
    selected_segment = st.sidebar.selectbox(
        "Click a segment from grid or choose manually",
        list(segment_counts.keys()), index=0
    )

    # ---------------- LEFT: RFM GRID ----------------
    with left:
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches

        colors = {
            "Recent Users": "#55efc4", "Promising": "#a29bfe", "About To Sleep": "#ff7675",
            "Hibernating": "#fab1a0", "Lost": "#e17055", "Price Sensitive": "#74b9ff",
            "Can't Lose Them": "#1B1464", "Need Attention": "#58B19F", "Loyal Customers": "#3D84F7",
            "Champions": "#00cec9", "Potential Loyalists": "#2475B0", "Other": "#CCCCCC"
        }

        fig, ax = plt.subplots(figsize=(9.5, 6.3))
        ax.set_xlim(1, 6); ax.set_ylim(1, 6)
        ax.invert_yaxis(); ax.axis('off')

        for segment, (x, y, w, h) in segment_positions.items():
            count = segment_counts.get(segment, 0)
            pct   = segment_percents.get(segment, 0)
            color = colors.get(segment, '#ccc')
            alpha = 0.95 if segment == selected_segment else 0.55

            ax.add_patch(patches.Rectangle((x, y), w, h,
                       facecolor=color, edgecolor='white', linewidth=2.5, alpha=alpha))
            cx = x + w / 2; cy = y + h / 2
            ax.text(cx, cy - 0.18, segment, ha='center', va='center',
                    fontsize=11, weight='bold', color='white')
            ax.text(cx, cy + 0.32, f"{count:,}  ({pct:.2f}%)",
                    ha='center', va='center', fontsize=10, color='white')

        for i in range(1, 6):
            ax.text(i + 0.5, 6.12, str(i), ha='center', va='top', fontsize=12)
            ax.text(0.88, i + 0.5, str(6 - i), ha='right', va='center', fontsize=12)

        ax.text(3.5, 6.45, "Recency score", ha='center', fontsize=14)
        ax.text(0.35, 3.5, "Frequency & Monetary score",
                rotation='vertical', va='center', fontsize=14)

        st.pyplot(fig)


    # ---------------- RIGHT: INFO & EXPORT ----------------
    with right:
        seg_df = rfm[rfm['Segment'] == selected_segment]
        num_customers     = len(seg_df)
        percent_customers = (num_customers / len(rfm) * 100) if len(rfm) else 0
        revenue           = seg_df['monetary'].sum()
        revenue_percent   = (revenue / total_revenue * 100) if total_revenue else 0
        churn_rate        = seg_df['churned'].mean() * 100 if 'churned' in seg_df.columns else None

        # Panel: current segment headline + KPIs
        st.markdown(f"""
        <div class="panel">
          <h3>Segment: {selected_segment}
            <span class="badge">{num_customers:,} customers • {percent_customers:.2f}%</span>
          </h3>
          <div class="kpi">
            <div><b>Revenue</b>: ${revenue:,.2f} <span class="soft">({revenue_percent:.2f}%)</span></div>
            {"<div><b>Churn rate</b>: " + f"{churn_rate:.2f}%" + "</div>" if churn_rate is not None else ""}
          </div>
          <div style="margin-top:10px">
            <small class="soft">Download the full list of customer IDs for this segment.</small><br/>
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.download_button(
            label="📥 Download Customer IDs",
            data=seg_df["Customer/ID"].to_csv(index=False),
            file_name=f"{selected_segment}_customer_ids.csv",
            mime="text/csv",
            use_container_width=True
        )

        st.divider()

        # -------- Export At‑Risk Customers (uses Churn tab params) --------
        st.markdown('<div class="panel"><h3>Export At‑Risk Customers in This Segment</h3>', unsafe_allow_html=True)

        bp = st.session_state.get("biz_params", {
            "clv": 300.0, "profit_margin": 0.2,
            "cost_employee_visit": 15.0, "cost_promo_discount": 20.0,
            "effectiveness_rate": 0.5, "cost_sms": 0.05,
            "cost_call": 0.2, "cost_other": 0.0
        })

        st.caption(
            f"Using Churn tab params → CLV ${bp['clv']:.0f}, margin {bp['profit_margin']:.0%}, "
            f"visit ${bp['cost_employee_visit']:.2f}, promo ${bp['cost_promo_discount']:.2f} × eff {bp['effectiveness_rate']:.0%}, "
            f"sms ${bp['cost_sms']:.2f}, call ${bp['cost_call']:.2f}, other ${bp['cost_other']:.2f}"
        )

        # choose best-cost model/threshold
        model_cols = [c for c in df.columns if c.endswith("_proba") and c != "true_label"]
        if not model_cols:
            st.warning("No *_proba columns found in df_ensemble_all_predictions.csv")
        else:
            thresholds = np.linspace(0.05, 0.95, 19)
            true_labels = df["true_label"].values if "true_label" in df.columns else None

            def total_cost_for(series, t):
                preds = (series >= t).astype(int)
                fp = int(((preds == 1) & (true_labels == 0)).sum()) if true_labels is not None else preds.sum()
                fn = int(((preds == 0) & (true_labels == 1)).sum()) if true_labels is not None else 0
                cost_fp = fp * (bp["cost_employee_visit"] + bp["cost_sms"] + bp["cost_call"] + bp["cost_other"] +
                                (bp["cost_promo_discount"] * bp["profit_margin"] * bp["effectiveness_rate"]))
                cost_fn = fn * bp["clv"] * bp["profit_margin"]
                return cost_fp + cost_fn

            best_model, best_thresh, best_cost = None, None, None
            for mc in model_cols:
                s = df[mc].values
                for t in thresholds:
                    cst = total_cost_for(s, t)
                    if (best_cost is None) or (cst < best_cost):
                        best_cost, best_model, best_thresh = cst, mc, float(t)

            # Compact override controls
            oc1, oc2 = st.columns([1,1])
            with oc1:
                override_model = st.selectbox("Model", ["(auto)"] + model_cols, index=0, key="seg_model_override")
            with oc2:
                override_thresh = st.slider("Threshold", 0.0, 1.0, best_thresh, step=0.01, key="seg_thresh_override")

            use_model  = best_model if override_model == "(auto)" else override_model
            use_thresh = override_thresh

            st.markdown(
                f"**Selected segment:** `{selected_segment}`  &nbsp;&nbsp; "
                f"**Best:** `{best_model}` @ `{best_thresh:.2f}` "
                f"<span class='badge'>Cost ${best_cost:,.2f}</span>",
                unsafe_allow_html=True
            )

            if use_model is None:
                st.info("Select a model to proceed.")
            else:
                preds   = (df[use_model] >= use_thresh).astype(int)
                at_risk = df.loc[preds == 1].copy()

                seg_map = rfm[["Customer/ID", "Segment"]]
                merged  = (at_risk.merge(seg_map, how="left", on="Customer/ID")
                                  .query("Segment == @selected_segment"))

                export_cols = ["Customer/ID", "Segment", use_model]
                export_df = merged[export_cols].sort_values(use_model, ascending=False)

                st.markdown(f"**At‑risk customers in `{selected_segment}`:** {len(export_df):,}")
                if len(export_df) == 0:
                    st.caption("No customers matched with the current model/threshold.")
                else:
                    st.download_button(
                        "📥 Download CSV",
                        data=export_df.to_csv(index=False),
                        file_name=f"at_risk_{selected_segment.replace(' ','_')}_{use_model}_th{use_thresh:.2f}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )


# ---------------------- DASHBOARD 3: COLLABORATIVE FILTERING ------------------
# COLLABORATIVE FILTERING TAB
elif dashboard_choice == "Collaborative Filtering":
    st.title("Engage - Collaborative Filtering Dashboard")

    items_df, segpop_df, sim_df = load_parquets()

    # ---- session basket ----
    if "basket_ids" not in st.session_state:
        st.session_state.basket_ids = []

    # ---- Add to basket: by ID (type-ahead "ID (Name)") or by Name ----
    st.subheader("Add Products to Basket")

    # Build display options like "ID (Name)" so typing matches both
    id_display = (items_df
                  .assign(display=items_df["product_id"] + " (" + items_df["product_name"] + ")")
                  .sort_values("display"))[["product_id", "product_name", "display"]]
    display_to_id = dict(zip(id_display["display"], id_display["product_id"]))
    name_options = [""] + items_df["product_name"].sort_values().tolist()
    id_options   = [""] + id_display["display"].tolist()

    c1, c2, c3, c4 = st.columns([1.8, 1.8, 0.9, 0.9])

    with c1:
        pid_choice_disp = st.selectbox(
            "Search by Product ID (type to filter)",
            options=id_options, index=0, key="add_by_id_disp"
        )
    with c2:
        pname_choice = st.selectbox(
            "Or pick by Name",
            options=name_options, index=0, key="add_by_name"
        )
    with c3:
        if st.button("➕ Add"):
            new_ids = []
            # from ID (ID (Name) → ID)
            if pid_choice_disp:
                new_ids.append(display_to_id[pid_choice_disp])
            # from Name → ID
            if pname_choice:
                new_ids += names_to_ids(items_df, [pname_choice])

            valid_ids = set(items_df["product_id"])
            for pid in new_ids:
                if pid in valid_ids and pid not in st.session_state.basket_ids:
                    st.session_state.basket_ids.append(pid)
    with c4:
        if st.button("🗑 Clear Basket"):
            st.session_state.basket_ids = []

    # Basket preview
    st.subheader("Current Basket")

    ids = st.session_state.basket_ids
    if not ids:
        st.info("Add items by ID or Name to build a basket.")
    else:
        # optional: collapse very long baskets
        with st.expander(f"{len(ids)} item(s) in basket", expanded=True):
            to_remove = []
            name_map = dict(zip(items_df["product_id"], items_df["product_name"]))

            for pid in ids:
                name = name_map.get(pid, pid)
                row = st.container()
                with row:
                    c1, c2 = st.columns([8,1])
                    with c1:
                        st.markdown(
                            f'<div class="card"><span class="item-name">{name}</span>'
                            f'<span class="item-id">({pid})</span></div>',
                            unsafe_allow_html=True
                        )
                    with c2:
                        if st.button("Remove", key=f"rm_{pid}"):
                            to_remove.append(pid)

            if to_remove:
                for pid in to_remove:
                    if pid in st.session_state.basket_ids:
                        st.session_state.basket_ids.remove(pid)
                st.rerun()
    st.divider()

    # ---- Three columns: CF / Segment / Random ----
    col1, spacer1, col2, spacer2, col3 = st.columns([3, 0.5, 3, 0.5, 3])

    basket_ids = st.session_state.basket_ids
    exclude = set(basket_ids)

    # ===== Column 1: CF =====
    with col1:
        k_cf = st.session_state.get("k_cf", 6)
        st.markdown(f"### CF Recommendations (Top {k_cf})")
        st.selectbox("How many?", [6, 12, 18], index=[6,12,18].index(k_cf), key="k_cf")
        k_cf = st.session_state["k_cf"]  # updated after rerun

        cf_ids = recommend_from_matrix(basket_ids, sim_df, k_cf) if basket_ids else []
        exclude.update(cf_ids)

        if not cf_ids:
            st.write("—")
        else:
            for pid, nm in zip(cf_ids, ids_to_names(items_df, cf_ids)):
                st.markdown(
                    f"""
                    <div style="
                        background:#E3F2FD;
                        padding:8px 10px;
                        border-radius:10px;
                        margin:6px 0;
                        border:1px solid #e6eefb;">
                        <b>{nm}</b> <span style="color:#64748b;">({pid})</span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    # ===== Column 2: Segment Picks =====
    with col2:
        k_seg = st.session_state.get("k_seg", 2)
        st.markdown(f"### Segment Picks (Top {k_seg})")
        st.selectbox("How many?", [2, 4, 6], index=[2,4,6].index(k_seg), key="k_seg")
        k_seg = st.session_state["k_seg"]

        seg_options = [""] + sorted(segpop_df["segment"].unique().tolist())
        chosen_segment = st.selectbox("Segment", options=seg_options, index=0, key="chosen_segment")

        seg_ids = segment_top_products(segpop_df, chosen_segment, k_seg, exclude)
        exclude.update(seg_ids)

        if not seg_ids:
            st.write("—")
        else:
            for pid, nm in zip(seg_ids, ids_to_names(items_df, seg_ids)):
                st.markdown(
                    f"""
                    <div style="
                        background:#E8F5E9;
                        padding:8px 10px;
                        border-radius:10px;
                        margin:6px 0;
                        border:1px solid #dff3e3;">
                        <b>{nm}</b> <span style="color:#64748b;">({pid})</span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    # ===== Column 3: Random (with Advanced Selection) =====
    with col3:
        k_rand = st.session_state.get("k_rand", 2)
        st.markdown(f"### Random (Top {k_rand})")
        st.selectbox("How many?", [2, 4, 6], index=[2,4,6].index(k_rand), key="k_rand")
        k_rand = st.session_state["k_rand"]

        # --- Advanced selection expander ---
        st.caption("Optional: constrain random picks by profit margin and stock.")
        with st.expander("Advanced selection (optional) — upload CSV", expanded=False):
            st.markdown("**Upload CSV with columns:** `ID, Profit_Margin, Stock`")
            st.caption("All columns are required. `ID` must match product IDs in your catalog.")
            st.code("ID,Profit_Margin,Stock\n12345,0.28,120\n98765,0.34,60\nA1001,0.15,300", language="csv")

            adv_file = st.file_uploader("Upload CSV", type=["csv"], key="adv_random_csv")
            min_profit = st.number_input("Minimum Profit_Margin", min_value=0.0, max_value=1.0, value=0.20, step=0.01, key="adv_min_profit")
            min_stock  = st.number_input("Minimum Stock", min_value=0, value=50, step=1, key="adv_min_stock")
            use_advanced = st.checkbox("Use advanced filters for random picks", value=False, key="adv_use_filters")

            eligible_ids = None
            if adv_file is not None:
                try:
                    adv_df = pd.read_csv(adv_file)
                    required_cols = {"ID", "Profit_Margin", "Stock"}
                    if not required_cols.issubset(set(adv_df.columns)):
                        st.error(f"CSV must contain columns: {sorted(list(required_cols))}")
                    else:
                        # Normalize types
                        adv_df["ID"] = adv_df["ID"].astype(str)
                        adv_df["Profit_Margin"] = pd.to_numeric(adv_df["Profit_Margin"], errors="coerce")
                        adv_df["Stock"] = pd.to_numeric(adv_df["Stock"], errors="coerce")
                        adv_df = adv_df.dropna(subset=["Profit_Margin", "Stock"])

                        # Filter by thresholds
                        filt = (adv_df["Profit_Margin"] >= min_profit) & (adv_df["Stock"] >= min_stock)
                        eligible_ids = set(adv_df.loc[filt, "ID"].astype(str))
                        st.success(f"Eligible products after filters: {len(eligible_ids):,}")
                except Exception as e:
                    st.error(f"Could not read CSV: {e}")

        # Build pool excluding basket + CF + Segment
        pool_mask = ~items_df["product_id"].isin(exclude)
        pool_ids = set(items_df.loc[pool_mask, "product_id"].tolist())

        # If using advanced and we have a valid eligible_ids set, intersect
        if use_advanced and eligible_ids is not None:
            filtered_pool = list(pool_ids & eligible_ids)
        else:
            filtered_pool = list(pool_ids)

        # Sample from filtered pool
        if not filtered_pool:
            rand_ids = []
        else:
            rng = np.random.default_rng(123)
            take = min(k_rand, len(filtered_pool))
            rand_ids = list(rng.choice(filtered_pool, size=take, replace=False))

        if not rand_ids:
            st.write("—")
        else:
            for pid, nm in zip(rand_ids, ids_to_names(items_df, rand_ids)):
                st.markdown(
                    f"""
                    <div style="
                        background:#FFFDE7;
                        padding:8px 10px;
                        border-radius:10px;
                        margin:6px 0;
                        border:1px solid #f3f0c6;">
                        <b>{nm}</b> <span style="color:#64748b;">({pid})</span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

