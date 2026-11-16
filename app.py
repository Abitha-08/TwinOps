import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from datetime import datetime
import requests

st.set_page_config(page_title="TwinOps | Software Optimization", page_icon="∞", layout="wide")

# ---------- Styling ----------
st.markdown("""
<style>
.main {background-color:#f5f7fb;}
.block-container {padding-top:1.2rem;}
[data-testid="stSidebar"] {background:#101936;}
[data-testid="stSidebar"] * {color:#f5f7ff !important;}
.metric-card {background:white;padding:18px;border-radius:14px;border:1px solid #e7eaf2;}
.small-muted {color:#64748b;font-size:13px;}
</style>
""", unsafe_allow_html=True)

# ---------- Demo data ----------
@st.cache_data
def make_history(seed=17):
    rng = np.random.default_rng(seed)
    n = 240
    cpu = np.clip(rng.normal(42, 10, n), 12, 75)
    memory = np.clip(rng.normal(56, 9, n), 25, 82)
    requests = np.clip(rng.normal(650, 150, n), 150, 1100)
    response = np.clip(0.35 + cpu/100*1.1 + requests/1000*0.35 + rng.normal(0, .12, n), .2, 2.5)
    errors = np.clip(rng.normal(1.0, .45, n), 0, 3)
    df = pd.DataFrame({"cpu":cpu,"memory":memory,"requests":requests,"response_time":response,"error_rate":errors})
    return df

FEATURES = ["cpu", "memory", "requests", "response_time", "error_rate"]

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "selected_scenario" not in st.session_state:
    st.session_state.selected_scenario = "High CPU + slow response"

# ---------- Login ----------
def login_page():
    left, right = st.columns([1.05, 1])
    with left:
        st.markdown("<div style='background:linear-gradient(145deg,#101936,#263d86);padding:48px 30px;border-radius:18px;color:white;min-height:420px'>", unsafe_allow_html=True)
        st.markdown("# ∞ TwinOps")
        st.markdown("### Smarter Monitoring.\n### Safer Software Optimization.")
        st.write("An application-independent SLM-powered Digital Twin framework for monitoring, analysis and safer optimization.")
        st.markdown("<br><br><h2>Monitor → Analyze → Simulate → Improve</h2>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with right:
        st.subheader("Welcome Back")
        st.caption("Sign in to continue to the TwinOps demo")
        email = st.text_input("Email address", value="demo@twinops.com")
        password = st.text_input("Password", type="password", value="twinops123")
        if st.button("Sign In", type="primary", use_container_width=True):
            if email == "demo@twinops.com" and password == "twinops123":
                st.session_state.logged_in = True
                st.rerun()
            else:
                st.error("Invalid demo credentials. Please use the credentials shown below.")
        st.info("Demo login: demo@twinops.com  |  Password: twinops123")
        st.caption("This is demonstration authentication only, not production-grade account security.")

# ---------- Sidebar ----------
def sidebar():
    st.sidebar.markdown("# ∞ TwinOps")
    st.sidebar.caption("Software performance intelligence")
    page = st.sidebar.radio("NAVIGATION", ["Dashboard", "Digital Twin Topology", "ML & SLM Analysis", "Simulation & Trust", "Logout"], index=0)
    st.sidebar.divider()
    st.sidebar.markdown("**System status**")
    st.sidebar.success("Demo data source connected")
    st.sidebar.success("ML engine ready")
    st.sidebar.warning("Ollama SLM optional")
    return page

# ---------- Dashboard ----------
def dashboard(history, current):
    st.title("TwinOps Overview")
    st.caption("Monitor the sample application and review its current performance.")
    health = max(0, min(100, 100 - max(0,current['cpu']-60)*0.55 - max(0,current['response_time']-1)*12 - current['error_rate']*2))
    cols = st.columns(4)
    cols[0].metric("Health Score", f"{health:.1f}%", "Demo estimate")
    cols[1].metric("Response Time", f"{current['response_time']*1000:.0f} ms")
    cols[2].metric("Error Rate", f"{current['error_rate']:.1f}%")
    cols[3].metric("CPU Usage", f"{current['cpu']:.0f}%")
    st.subheader("Performance trend")
    plot_df = history.tail(36).reset_index(names="sample")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=plot_df["sample"], y=plot_df["cpu"], name="CPU %", mode="lines"))

    fig.add_trace(go.Scatter(x=plot_df["sample"], y=plot_df["memory"], name="Memory %", mode="lines"))

    fig.add_trace(go.Scatter(x=plot_df["sample"], y=plot_df["response_time"] * 30, name="Response time × 30", mode="lines"))
    fig.update_layout(height=340, margin=dict(l=10,r=10,t=20,b=10), legend=dict(orientation="h"))
    st.plotly_chart(fig, use_container_width=True)
    a,b = st.columns(2)
    with a:
        st.subheader("System status")
        st.success("Sample application connected")
        st.success("Digital Twin model loaded")
        st.success("ML analysis available")
        st.info("SLM status depends on local Ollama availability")
    with b:
        st.subheader("Latest metrics")
        st.dataframe(pd.DataFrame({"Metric":["CPU","Memory","Requests/min","Response time","Error rate"],"Value":[f"{current['cpu']:.1f}%",f"{current['memory']:.1f}%",f"{current['requests']:.0f}",f"{current['response_time']:.2f} sec",f"{current['error_rate']:.1f}%"]}), hide_index=True, use_container_width=True)

# ---------- Digital twin topology ----------
def topology(current):
    st.title("Digital Twin Topology")
    st.caption("A simplified virtual representation of the sample web application architecture.")
    st.info("This is a conceptual simulation of the application layers. It is not a live copy of an external website.")
    c1,c2,c3,c4 = st.columns(4)
    for col, name, detail in [(c1,"Users","Incoming requests"),(c2,"Web Layer","HTTP / API entry"),(c3,"Service Layer","Business logic"),(c4,"Database","Data access")]:
        with col:
            st.markdown(f"<div style='background:white;border:1px solid #dce3ef;border-radius:12px;padding:18px;text-align:center;min-height:112px'><h3>{name}</h3><span style='color:#64748b'>{detail}</span></div>", unsafe_allow_html=True)
    st.markdown("<div style='text-align:center;font-size:26px;color:#5266d8'>↓　↓　↓</div>", unsafe_allow_html=True)
    st.subheader("Component details – Web layer")
    x,y,z = st.columns(3)
    x.metric("Status", "Healthy" if current['cpu'] < 85 else "High load")
    y.metric("Response time", f"{current['response_time']*1000:.0f} ms")
    z.metric("CPU usage", f"{current['cpu']:.0f}%")
    st.subheader("Runtime metrics")
    st.dataframe(pd.DataFrame({"Component":["Web Layer","Service Layer","Database"],"Status":["Monitored","Simulated","Simulated"],"Observed information":[f"CPU {current['cpu']:.1f}% / response {current['response_time']:.2f}s","Request processing layer","Connection pool represented in simulation"]}),hide_index=True,use_container_width=True)

# ---------- ML + SLM ----------
def get_scenario():
    scenarios = {
        "Normal operation": [43,55,620,0.9,1.0],
        "High CPU + slow response": [95,85,1050,8.0,12.0],
        "High memory": [52,97,700,1.8,2.0],
        "High error rate": [68,72,820,3.2,18.0],
    }
    return scenarios[st.session_state.selected_scenario]

def ml_slm_page(history):
    st.title("Analysis Center")
    st.caption("Detect unusual metrics, estimate upcoming CPU load, and request an SLM explanation.")
    tabs = st.tabs(["Anomaly Detection", "Prediction", "SLM Analysis"])
    with tabs[0]:
        st.subheader("Test a performance scenario")
        st.selectbox("Choose sample scenario", ["Normal operation","High CPU + slow response","High memory","High error rate"], key="selected_scenario")
        values = get_scenario()
        sample = pd.DataFrame([values], columns=FEATURES)
        model = IsolationForest(contamination=0.05, random_state=42)
        model.fit(history[FEATURES])
        prediction = int(model.predict(sample)[0])
        score = float(model.decision_function(sample)[0])
        k = st.columns(5)
        labels = ["CPU %","Memory %","Requests/min","Response (sec)","Error %"]
        for col,label,val in zip(k,labels,values): col.metric(label, f"{val:.1f}")
        if prediction == -1:
            st.error("Anomaly detected: this scenario differs from the learned baseline pattern.")
        else:
            st.success("No anomaly detected: this scenario is within the model's learned baseline range.")
        st.caption(f"Isolation Forest decision score: {score:.4f} (model score, not a probability/confidence percentage).")
        st.write("Training baseline: 240 generated normal-operation records. Replace this with real monitored data for a real deployment.")
    with tabs[1]:
        st.subheader("CPU performance prediction")
        df = history.copy()
        # A simple demonstration target: next-step CPU approximated by the next row.
        df["target_cpu"] = df.cpu.shift(-1)
        train = df.dropna()
        X = train[["cpu","memory","requests","response_time","error_rate"]]
        y = train["target_cpu"]
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.2,random_state=42)
        reg = RandomForestRegressor(n_estimators=120, random_state=42)
        reg.fit(X_train,y_train)
        mae = mean_absolute_error(y_test, reg.predict(X_test))
        latest = history.iloc[-1:][FEATURES]
        predicted = float(reg.predict(latest)[0])
        p1,p2 = st.columns(2)
        p1.metric("Next-sample CPU estimate", f"{predicted:.1f}%")
        p2.metric("Test MAE", f"{mae:.1f} percentage points")
        st.caption("Demonstration model trained on generated sequential samples. It is not a validated production forecast.")
        chart = history.tail(30).reset_index(names="sample")
        fig = px.line(chart, x="sample", y="cpu", title="Recent CPU history")
        fig.add_hline(y=predicted, line_dash="dash", annotation_text=f"Next estimate {predicted:.1f}%")
        st.plotly_chart(fig,use_container_width=True)
    with tabs[2]:
        st.subheader("SLM insight")
        vals = get_scenario()
        prompt = f"You are a cautious software performance assistant. Analyze these metrics: CPU={vals[0]}%, memory={vals[1]}%, requests/min={vals[2]}, response_time={vals[3]} seconds, error_rate={vals[4]}%. Give possible causes (not certain claims) and 3 safe investigation or optimization steps. Do not claim to have changed the real application."
        st.code(prompt, language="text")
        model_name = st.text_input("Ollama model name", value="llama3.2")
        if st.button("Generate SLM analysis", type="primary"):
            try:
                response = requests.post("http://localhost:11434/api/generate", json={"model":model_name,"prompt":prompt,"stream":False}, timeout=90)
                response.raise_for_status()
                st.markdown("**Ollama SLM response**")
                st.write(response.json().get("response","No response text returned."))
            except Exception:
                st.warning("Ollama was not reachable. Showing a clearly labelled fallback explanation instead.")
                if vals[0] > 85 or vals[3] > 3:
                    st.markdown("**Possible interpretation:** High CPU and/or slow response may indicate request overload, inefficient application work, or database/network delays. These are hypotheses and should be checked against logs and traces.")
                    st.markdown("**Suggested checks:** 1. Inspect slow API/database queries. 2. Check request concurrency and worker saturation. 3. Review CPU and memory trends before changing resource limits.")
                elif vals[1] > 90:
                    st.markdown("**Possible interpretation:** High memory use may indicate large caches, memory leaks, or high concurrent workload. Check memory trend and garbage collection/logs.")
                else:
                    st.markdown("**Possible interpretation:** The selected sample is close to the generated baseline. Continue monitoring and compare with real workload history.")

# ---------- Simulation ----------
def simulation_page():
    st.title("Simulation Results")
    st.caption("What-if simulation for a proposed database connection-pool adjustment.")
    st.warning("Illustrative digital-twin calculation only. The simulation does not change a real application or prove the change will work in production.")
    st.subheader("Proposed action")
    pool = st.slider("Proposed maximum database connections", min_value=10, max_value=60, value=30, step=5)
    baseline_pool = 20
    st.write(f"Current pool size: **{baseline_pool}** → Proposed pool size: **{pool}**")
    # Transparent, simple illustrative model: benefits taper and resource cost rises.
    improvement = max(-15, min(35, (pool-baseline_pool)*1.05))
    response_before = 1200
    response_after = max(500, response_before*(1-improvement/100))
    error_before = 0.8
    error_after = max(0.1, error_before*(1-max(0,improvement)/100))
    trust = max(55, min(96, 90 - abs(pool-30)*0.8))
    a,b,c = st.columns(3)
    a.metric("Estimated response time", f"{response_after:.0f} ms", f"{(response_after/response_before-1)*100:.1f}%")
    b.metric("Estimated error rate", f"{error_after:.2f}%", f"{error_after-error_before:.2f} pp")
    c.metric("Simulation trust score", f"{trust:.1f}/100", "Illustrative score")
    st.subheader("Before vs simulated outcome")
    compare = pd.DataFrame({"Metric":["Response time (ms)","Error rate (%)","Connection pool"],"Current":[response_before,error_before,baseline_pool],"Simulated":[response_after,error_after,pool]})
    st.dataframe(compare,hide_index=True,use_container_width=True)
    st.info("Review the assumptions and validate with load testing before applying any configuration to a real system.")
    if st.button("Mark simulation as approved for review", type="primary"):
        st.success("Simulation marked as approved for human review. No live system configuration was changed.")

# ---------- Main ----------
history = make_history()
if not st.session_state.logged_in:
    login_page()
else:
    page = sidebar()
    # Default current observation is a normal sample; analysis page offers selectable stress scenarios.
    current = {"cpu":43.0,"memory":55.0,"requests":620.0,"response_time":0.9,"error_rate":1.0}
    if page == "Dashboard": dashboard(history,current)
    elif page == "Digital Twin Topology": topology(current)
    elif page == "ML & SLM Analysis": ml_slm_page(history)
    elif page == "Simulation & Trust": simulation_page()
    elif page == "Logout":
        st.session_state.logged_in = False
        st.rerun()
