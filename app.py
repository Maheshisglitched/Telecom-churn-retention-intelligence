"""Telecom Churn Intelligence — Streamlit dashboard.
Keep churn_pipeline.joblib, model_report.json and customer_risk_predictions.csv beside this file.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="Telecom Churn Intelligence", page_icon="📡", layout="wide", initial_sidebar_state="expanded")

COLORS = {"Low": "#27d9b4", "Medium": "#f5ba55", "High": "#ff637d"}
TEMPLATE = "plotly_dark"
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');
html,body,[class*="css"],.stApp {font-family:'DM Sans',sans-serif;}
.stApp {background:radial-gradient(ellipse at 85% 0%,#19264b 0%,#0c1325 37%,#090f1c 78%);color:#e9f0ff;}
section[data-testid="stSidebar"] {background:#101a30;border-right:1px solid #263452;}
.block-container {padding-top:2rem;max-width:1450px;}
div[data-testid="stMetric"] {background:linear-gradient(135deg,#172541,#101a30);border:1px solid #2a3b5e;border-radius:17px;padding:18px 20px;}
div[data-testid="stMetricLabel"] {color:#9cb2d4;}
div[data-testid="stMetricValue"] {color:#f2f7ff;}
div[data-testid="stVerticalBlockBorderWrapper"] {border-color:#283955;}
.hero {padding:26px 30px;border-radius:22px;background:linear-gradient(110deg,#263f82,#37266d 65%,#192b4d);border:1px solid #485b92;margin-bottom:20px;}
.hero h1 {font-size:2.15rem;margin:0 0 8px 0;color:white;}
.hero p {color:#c5d6f3;margin:0;}
.eyebrow {font-size:12px;font-weight:700;letter-spacing:2px;color:#70d7ff;margin-bottom:8px;}
.note {color:#a5b5d0;font-size:13px;}
.risk {padding:7px 13px;border-radius:999px;font-weight:700;display:inline-block;}
</style>""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return joblib.load(ROOT / "churn_pipeline.joblib")

@st.cache_data
def load_report():
    return json.loads((ROOT / "model_report.json").read_text(encoding="utf-8"))

@st.cache_data
def load_portfolio():
    p = ROOT / "customer_risk_predictions.csv"
    return pd.read_csv(p) if p.exists() else None

try:
    model, report = load_model(), load_report()
    required = list(report["feature_columns"])
except Exception as exc:
    st.error(f"Unable to load trained model or report: {exc}")
    st.info("Place churn_pipeline.joblib and model_report.json alongside app.py and use your Python 3.13 environment with scikit-learn 1.6.1.")
    st.stop()


def score(df):
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))
    features = df[required].copy()
    if "TotalCharges" in features:
        features["TotalCharges"] = pd.to_numeric(features["TotalCharges"], errors="coerce")
    prob = model.predict_proba(features)[:, 1]
    result = df.copy()
    result["churn_probability"] = prob
    result["risk_segment"] = pd.cut(prob, [-0.001, 0.30, 0.60, 1.001], labels=["Low", "Medium", "High"], right=False)
    return result


def actions(row):
    rec = []
    if row.get("Contract") == "Month-to-month":
        rec.append("Explore an optional longer-term plan with transparent pricing.")
    if row.get("InternetService") == "Fiber optic" and row.get("TechSupport") == "No":
        rec.append("Offer a proactive service and technical-support check-in.")
    if row.get("PaymentMethod") == "Electronic check":
        rec.append("Explain available payment options and autopay benefits.")
    tenure = pd.to_numeric(row.get("tenure"), errors="coerce")
    if pd.notna(tenure) and tenure < 12:
        rec.append("Schedule a new-customer satisfaction and onboarding review.")
    return rec or ["Ask about customer needs before considering a personalized retention offer."]


def hero(kicker, title, description):
    st.markdown(f'<div class="hero"><div class="eyebrow">{kicker}</div><h1>{title}</h1><p>{description}</p></div>', unsafe_allow_html=True)


def risk_chart(scored):
    counts = scored["risk_segment"].value_counts().reindex(["Low", "Medium", "High"], fill_value=0)
    fig = go.Figure(go.Pie(labels=counts.index.tolist(), values=counts.values, hole=.7,
                           marker=dict(colors=[COLORS[k] for k in counts.index]),
                           textinfo="label+percent", sort=False))
    fig.update_layout(template=TEMPLATE, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      margin=dict(t=15,b=15,l=10,r=10), height=330, showlegend=False,
                      annotations=[dict(text=f"{len(scored):,}<br><span style='font-size:12px'>customers</span>",x=.5,y=.5,showarrow=False,font_size=25)])
    return fig


def gauge(prob):
    color = COLORS["High" if prob >= .6 else "Medium" if prob >= .3 else "Low"]
    fig = go.Figure(go.Indicator(mode="gauge+number",value=prob*100,number={"suffix":"%","font":{"size":46}},
        gauge={"axis":{"range":[0,100]},"bar":{"color":color,"thickness":.3},
               "bgcolor":"#1b2b49","steps":[{"range":[0,30],"color":"#163c3c"},{"range":[30,60],"color":"#554126"},{"range":[60,100],"color":"#54283c"}]}))
    fig.update_layout(template=TEMPLATE,paper_bgcolor="rgba(0,0,0,0)",height=260,margin=dict(t=25,b=5,l=25,r=25))
    return fig


def customer_form(source):
    defaults = source.iloc[0]
    with st.form("predict_one"):
        values = {}
        cols = st.columns(3)
        for i, name in enumerate(required):
            with cols[i % 3]:
                s = source[name]
                if name in ("tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"):
                    num = pd.to_numeric(s,errors="coerce")
                    fallback = float(num.median()) if num.notna().any() else 0.0
                    init = pd.to_numeric(defaults[name],errors="coerce")
                    init = float(init) if pd.notna(init) else fallback
                    if name in ("tenure", "SeniorCitizen"):
                        values[name] = st.number_input(name,min_value=0,value=max(0,int(init)),step=1)
                    else:
                        values[name] = st.number_input(name,min_value=0.0,value=max(0.,init),step=1.0)
                else:
                    options = sorted(s.dropna().astype(str).unique().tolist())
                    if not options:
                        options = [""]
                    initial = str(defaults[name])
                    values[name] = st.selectbox(name,options,index=options.index(initial) if initial in options else 0)
        submitted = st.form_submit_button("🔮 Predict customer churn",type="primary",use_container_width=True)
    return values if submitted else None


def explain_one(values, source):
    """Compute actual SHAP values for a supported linear pipeline, using a sample of transformed portfolio as background."""
    import shap
    from scipy import sparse
    prep = model[:-1]
    estimator = model.steps[-1][1]
    if not hasattr(estimator, "coef_"):
        raise ValueError("Individual SHAP explanations currently require a linear classifier.")
    background = source[required].sample(n=min(120,len(source)),random_state=42).copy()
    selected = pd.DataFrame([values])[required].copy()
    if "TotalCharges" in required:
        background["TotalCharges"] = pd.to_numeric(background["TotalCharges"],errors="coerce")
        selected["TotalCharges"] = pd.to_numeric(selected["TotalCharges"],errors="coerce")
    bg = prep.transform(background)
    x = prep.transform(selected)
    if sparse.issparse(bg):
        bg = bg.toarray()
    if sparse.issparse(x):
        x = x.toarray()
    names = list(prep.get_feature_names_out())
    explainer = shap.LinearExplainer(estimator, bg)
    explanation = explainer(x)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9,5.5))
    shap.plots.waterfall(shap.Explanation(values=explanation.values[0],
                       base_values=float(np.ravel(explanation.base_values)[0]),
                       data=x[0],feature_names=names),max_display=12,show=False)
    fig = plt.gcf()
    st.pyplot(fig,clear_figure=True)
    plt.close(fig)
    st.caption("SHAP values explain changes in model log-odds, not percentage-point changes in churn probability. The background is a sample of the supplied portfolio.")


with st.sidebar:
    st.markdown("### 📡 CHURN LAB")
    st.caption("CUSTOMER INTELLIGENCE SUITE")
    st.divider()
    page = st.radio("Navigate",["Overview","Churn Prediction","Explainable AI","Retention Intelligence","Batch Predictions"],label_visibility="collapsed")
    st.divider()
    st.caption("🟢 Local model connected")
    st.caption("Built with Streamlit · Scikit-learn")
    st.caption("Educational prototype · Validate before business use")

source = load_portfolio()
scored = None
if source is not None:
    try:
        scored = score(source)
    except Exception as exc:
        st.warning(f"Saved portfolio cannot be scored: {exc}")

if page == "Overview":
    hero("PORTFOLIO OVERVIEW", "Your customers, at a glance", "Explore churn exposure, customer segments and patterns across your portfolio.")
    if scored is None:
        st.info("Add customer_risk_predictions.csv beside app.py to unlock portfolio analytics.")
    else:
        high = scored["risk_segment"].eq("High")
        a,b,c,d = st.columns(4)
        a.metric("👥 Total customers",f"{len(scored):,}")
        b.metric("🚨 High-risk customers",f"{high.sum():,}")
        c.metric("📈 Average churn risk",f"{scored['churn_probability'].mean():.1%}")
        d.metric("🎯 High-risk share",f"{high.mean():.1%}")
        left,right = st.columns([1,1.35])
        with left:
            st.subheader("Risk mix")
            st.plotly_chart(risk_chart(scored),use_container_width=True)
        with right:
            st.subheader("Churn risk by contract")
            if "Contract" in scored:
                group = scored.groupby("Contract",dropna=False)["churn_probability"].mean().mul(100).reset_index()
                fig = px.bar(group,x="Contract",y="churn_probability",color="Contract",text_auto=".1f",template=TEMPLATE,
                             color_discrete_sequence=["#68b8ff","#a78bfa","#2dd4bf"])
                fig.update_layout(showlegend=False,paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
                                  yaxis_title="Average predicted churn (%)",height=330,margin=dict(t=15,b=15))
                st.plotly_chart(fig,use_container_width=True)
        st.subheader("Customer risk explorer")
        f1,f2 = st.columns([1,2])
        with f1:
            filter_risk = st.multiselect("Risk categories",["High","Medium","Low"],default=["High","Medium","Low"])
        with f2:
            query = st.text_input("Search customer ID",placeholder="Enter all or part of a customer ID")
        view = scored[scored["risk_segment"].astype(str).isin(filter_risk)].copy()
        if query and "customerID" in view:
            view = view[view["customerID"].astype(str).str.contains(query,case=False,regex=False)]
        st.dataframe(view.sort_values("churn_probability",ascending=False).head(200),use_container_width=True,hide_index=True,
                     column_config={"churn_probability":st.column_config.ProgressColumn("Churn probability",min_value=0,max_value=1,format="%.1f")})
        st.caption("Risk scores are model estimates, not observed future churn. Showing up to 200 matching records.")

elif page == "Churn Prediction":
    hero("PREDICTIVE ENGINE", "Predict a customer's churn risk", "Enter customer details to generate a risk score and tailored follow-up ideas.")
    if source is None:
        st.warning("Add customer_risk_predictions.csv to populate the form with valid categories.")
    else:
        with st.expander("📝 Customer information",expanded=True):
            values = customer_form(source)
        if values is not None:
            try:
                result = score(pd.DataFrame([values]))
                prob = float(result["churn_probability"].iloc[0])
                risk = str(result["risk_segment"].iloc[0])
                st.session_state["last_customer"] = values
                a,b = st.columns([1,1.3])
                with a:
                    st.plotly_chart(gauge(prob),use_container_width=True)
                with b:
                    st.subheader(f"{risk} risk customer")
                    st.write("**Suggested next steps**")
                    for action in actions(values):
                        st.write("• "+action)
                    st.caption("Low <30%, Medium 30–<60%, High ≥60%. These suggestions are rules, not proven interventions.")
                st.success("Prediction complete. Visit Explainable AI to investigate this customer's prediction.")
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")

elif page == "Explainable AI":
    hero("MODEL TRANSPARENCY", "Understand what drives churn", "Explore the model's strongest signals and explain an individual prediction with SHAP.")
    st.subheader("Top model coefficients")
    try:
        prep = model[:-1]
        estimator = model.steps[-1][1]
        names = prep.get_feature_names_out()
        coef = estimator.coef_[0]
        imp = pd.DataFrame({"feature":names,"coefficient":coef})
        imp = imp.reindex(imp.coefficient.abs().sort_values(ascending=False).index).head(16).sort_values("coefficient")
        fig = px.bar(imp,x="coefficient",y="feature",orientation="h",color="coefficient",color_continuous_scale=["#28d9b5","#7c84d6","#ff6382"],template=TEMPLATE)
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",height=520,coloraxis_showscale=False,
                          margin=dict(t=15,b=15,l=10,r=15),yaxis_title="",xaxis_title="Coefficient (log-odds)")
        st.plotly_chart(fig,use_container_width=True)
        st.caption("Coefficients reflect associations in encoded features and are not causal effects. Their magnitudes depend on feature scaling.")
    except Exception as exc:
        st.info(f"Coefficient chart unavailable for this model: {exc}")
    st.subheader("🔍 Explain one prediction with SHAP")
    if source is None:
        st.info("Add customer_risk_predictions.csv to enable individual SHAP explanations.")
    else:
        mode = st.radio("Choose a customer",["Last prediction","Saved portfolio row"],horizontal=True)
        selected = None
        if mode == "Last prediction":
            selected = st.session_state.get("last_customer")
            if selected is None:
                st.info("First submit a prediction on the Churn Prediction page, or select a saved portfolio row.")
        else:
            idx = st.number_input("Portfolio row (starting at 1)",min_value=1,max_value=len(source),value=1,step=1)
            selected = source.iloc[int(idx)-1][required].to_dict()
        if selected is not None and st.button("✨ Generate SHAP explanation",type="primary"):
            try:
                explain_one(selected,source)
            except ImportError:
                st.warning("SHAP is not installed. Run: python -m pip install shap matplotlib")
            except Exception as exc:
                st.warning(f"SHAP explanation unavailable: {exc}")
    with st.expander("Model evaluation report"):
        st.json({k:v for k,v in report.items() if k!="feature_columns"})

elif page == "Retention Intelligence":
    hero("RETENTION WORKSPACE", "Turn predictions into priorities", "Review customers who may need attention and explore rule-based outreach ideas.")
    if scored is None:
        st.info("Add customer_risk_predictions.csv to view the retention queue.")
    else:
        risk = st.segmented_control("Customer risk segment",["High","Medium","Low","All"],default="High")
        selected = scored if risk=="All" else scored[scored["risk_segment"].astype(str)==risk]
        a,b = st.columns(2)
        a.metric("Customers in queue",f"{len(selected):,}")
        b.metric("Average predicted risk",f"{selected['churn_probability'].mean():.1%}" if len(selected) else "—")
        st.dataframe(selected.sort_values("churn_probability",ascending=False).head(300),use_container_width=True,hide_index=True)
        if len(selected):
            st.subheader("Customer action plan")
            indices = selected.index.tolist()
            idx = st.selectbox("Select customer",indices,format_func=lambda i:f"{selected.loc[i].get('customerID','Customer')} · row {i+1}")
            row = selected.loc[idx]
            st.plotly_chart(gauge(float(row["churn_probability"])),use_container_width=True)
            for action in actions(row):
                st.write("• "+action)
        st.caption("These suggestions are heuristic. Confirm customer preferences, eligibility, and business policy before contacting anyone.")

elif page == "Batch Predictions":
    hero("BATCH SCORING", "Score your customer portfolio", "Upload a CSV, analyze predicted churn risk and export scored results.")
    upload = st.file_uploader("Upload customer CSV",type=["csv"])
    with st.expander("Required model input columns"):
        st.write(", ".join(required))
    if upload is not None:
        try:
            df = pd.read_csv(upload)
            if df.empty:
                st.warning("The uploaded CSV contains no customer rows.")
            else:
                out = score(df)
                a,b,c = st.columns(3)
                a.metric("Customers scored",f"{len(out):,}")
                b.metric("High risk",f"{out['risk_segment'].eq('High').sum():,}")
                c.metric("Average churn risk",f"{out['churn_probability'].mean():.1%}")
                st.plotly_chart(risk_chart(out),use_container_width=True)
                st.dataframe(out.sort_values("churn_probability",ascending=False),use_container_width=True,hide_index=True)
                st.download_button("⬇ Download scored CSV",out.to_csv(index=False).encode("utf-8"),file_name="scored_customers.csv",mime="text/csv",type="primary")
        except Exception as exc:
            st.error(f"Unable to score uploaded CSV: {exc}")
    else:
        st.info("Upload a CSV with the required features. Do not upload personal customer information without authorization.")
