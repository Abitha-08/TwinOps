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

st.set_page_config(
    page_title="TwinOps | Software Optimization",
    page_icon="∞",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- Styling ----------

# ---------- Styling ----------

# ---------- Styling ----------
st.markdown("""
<style>

/* Main application background */
.stApp,
[data-testid="stAppViewContainer"] {
    background: #F5F8FF !important;
    color: #172554 !important;
}

/* Full-width responsive content */
.block-container {
    max-width: 100% !important;
    padding: 2rem 2.5rem !important;
}

/* Sidebar size and background */
section[data-testid="stSidebar"] {
    width: 320px !important;
    min-width: 320px !important;
}

section[data-testid="stSidebar"] > div {
    width: 320px !important;
    background: linear-gradient(180deg, #0B1735, #142D58) !important;
}

/* Sidebar typography */
section[data-testid="stSidebar"] h1 {
    color: #FFFFFF !important;
    font-size: 30px !important;
    font-weight: 800 !important;
}

section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] span {
    color: #F1F5FF !important;
}

/* Navigation heading */
section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] {
    font-size: 15px !important;
    font-weight: 700 !important;
}

/* Navigation spacing */
section[data-testid="stSidebar"] div[role="radiogroup"] {
    gap: 8px !important;
}

section[data-testid="stSidebar"] div[role="radiogroup"] label {
    min-height: 42px !important;
    padding: 9px 10px !important;
    border-radius: 8px !important;
    font-size: 15px !important;
    font-weight: 600 !important;
}

/* Sidebar dividers */
section[data-testid="stSidebar"] hr {
    border-color: #34496F !important;
}

/* Main headings */
h1, h2, h3 {
    color: #14213D !important;
}

/* Buttons */
.stButton > button {
    background: #1764E8 !important;
    color: white !important;
    border: none !important;
    border-radius: 9px !important;
    padding: 10px 18px !important;
}

/* Metric cards */
.metric-card {
    background: white;
    padding: 20px;
    border-radius: 14px;
    border: 1px solid #DCE5F5;
}

/* Responsive laptop and tablet layout */
@media (max-width: 1000px) {
    section[data-testid="stSidebar"] {
        width: 280px !important;
        min-width: 280px !important;
    }

    section[data-testid="stSidebar"] > div {
        width: 280px !important;
    }

    .block-container {
        padding: 1.2rem !important;
    }
}

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


    # ---------- Session State ----------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "user_role" not in st.session_state:
    st.session_state.user_role = None

if "current_user" not in st.session_state:
    st.session_state.current_user = None

if "selected_scenario" not in st.session_state:
    st.session_state.selected_scenario = "High CPU + slow response"

# ---------- Login ----------

    
# ---------- Login ----------

# ---------- Login ----------
def login_page():

    st.markdown("""
    <style>
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
        max-width: 100% !important;
    }

    .login-brand {
        min-height: 650px;
        height: 100%;
        background: linear-gradient(140deg, #09152F, #142D58, #284B91);
        border-radius: 24px;
        padding: 65px 55px;
        color: white;
        display: flex;
        flex-direction: column;
        justify-content: center;
        box-shadow: 0 15px 40px rgba(15, 35, 75, 0.18);
    }

    .login-brand h1 {
        color: white !important;
        font-size: 55px !important;
        font-weight: 850 !important;
        margin-bottom: 20px;
    }

    .login-brand h2 {
        color: white !important;
        font-size: 32px !important;
        line-height: 1.4;
    }

    .login-brand p {
        color: #DCE8FF !important;
        font-size: 19px;
        line-height: 1.8;
    }

    .login-brand .tagline {
        color: #91B8FF;
        font-size: 17px;
        font-weight: 700;
        letter-spacing: 2px;
        margin-bottom: 25px;
    }

    div[data-testid="stTextInput"] input {
        min-height: 52px;
        font-size: 17px;
        border-radius: 9px;
    }

    div[data-testid="stSelectbox"] div[data-baseweb="select"] {
        min-height: 52px;
    }

    .login-heading {
        font-size: 38px;
        font-weight: 800;
        color: #14213D;
        margin-bottom: 5px;
    }

    .login-subtitle {
        font-size: 17px;
        color: #64748B;
        margin-bottom: 30px;
    }

    @media (max-width: 800px) {
        .login-brand {
            min-height: 350px;
            padding: 35px 25px;
        }

        .login-brand h1 {
            font-size: 38px !important;
        }

        .login-brand h2 {
            font-size: 24px !important;
        }

        .block-container {
            padding: 1rem !important;
        }
    }

/* Sidebar navigation design */
section[data-testid="stSidebar"] {
    min-width: 290px !important;
    width: 290px !important;
}

section[data-testid="stSidebar"] > div {
    padding: 25px 18px;
}

section[data-testid="stSidebar"] h1 {
    font-size: 30px !important;
    font-weight: 850 !important;
}

section[data-testid="stSidebar"] [data-testid="stRadio"] label {
    font-size: 16px !important;
    font-weight: 600 !important;
    padding: 9px 5px !important;
}

section[data-testid="stSidebar"] [data-testid="stRadio"] {
    gap: 10px !important;
}

section[data-testid="stSidebar"] [data-testid="stAlert"] {
    border-radius: 12px;
}

section[data-testid="stSidebar"] hr {
    margin-top: 18px;
    margin-bottom: 18px;
}

    </style>
    """, unsafe_allow_html=True)

    left, right = st.columns(
        [1.2, 1],
        gap="large",
        vertical_alignment="center"
    )

    with left:
        st.markdown("""
        <div class="login-brand">
            <div class="tagline">SOFTWARE PERFORMANCE INTELLIGENCE</div>
            <h1>∞ TwinOps</h1>
            <h2>Smarter Monitoring.<br>Safer Software Optimization.</h2>
            <p>
                An application-independent SLM-powered Digital Twin
                framework for monitoring, analysis and safer optimization.
            </p>
            <br>
            <h2>Monitor → Analyze<br>Simulate → Improve</h2>
            <br>
            <p>Intelligent insights for better software performance.</p>
        </div>
        """, unsafe_allow_html=True)

    with right:
        st.write("")
        st.write("")

        with st.container(border=True):
            st.markdown(
                '<div class="login-heading">Welcome Back</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                '<div class="login-subtitle">Sign in to continue to TwinOps</div>',
                unsafe_allow_html=True
            )

            role = st.selectbox(
                "Select Your Role",
                ["Admin", "Developer"]
            )

            email = st.text_input(
                "Email Address",
                placeholder="Enter your email address"
            )

            password = st.text_input(
                "Password",
                type="password",
                placeholder="Enter your password"
            )

            st.write("")

            if st.button(
                "Sign In",
                type="primary",
                use_container_width=True
            ):

                if role == "Admin":
                    if email == "admin@twinops.com" and password == "admin123":
                        st.session_state.logged_in = True
                        st.session_state.user_role = "Admin"
                        st.session_state.current_user = email
                        st.rerun()
                    else:
                        st.error("Invalid Admin credentials.")

                elif role == "Developer":
                    if email == "developer@twinops.com" and password == "dev123":
                        st.session_state.logged_in = True
                        st.session_state.user_role = "Developer"
                        st.session_state.current_user = email
                        st.rerun()
                    else:
                        st.error("Invalid Developer credentials.")

            st.divider()

            st.caption("DEMO LOGIN CREDENTIALS")

            st.info(
                "Admin: admin@twinops.com | Password: admin123\n\n"
                "Developer: developer@twinops.com | Password: dev123"
            )

            st.caption("Demonstration authentication only.")



# ---------- Sidebar ----------

# ---------- Sidebar ----------

# ---------- Sidebar ----------

# ---------- Sidebar ----------
def sidebar():

    st.sidebar.markdown("# ∞ TwinOps")
    st.sidebar.caption("SOFTWARE PERFORMANCE INTELLIGENCE")
    st.sidebar.divider()

    if st.session_state.user_role == "Admin":

        navigation = [
            "Dashboard",
            "Admin Panel",
            "Digital Twin Topology",
            "ML & SLM Analysis",
            "Simulation & Trust",
            "06 - Adaptation Centre",
            "07 - Alerts & Notifications",
            "08 - Performance Trends & Analytics",
            "09 - Knowledge Store & History",
            "10 - Settings & Configuration",
            "11 - Connector Management",
            "12 - E-Commerce Application",
            "13 - Reports & Export",
            "14 - About & Help",
            "Logout"
        ]

    else:

        navigation = [
            "Dashboard",
            "Digital Twin Topology",
            "ML & SLM Analysis",
            "Simulation & Trust",
            "06 - Adaptation Centre",
            "07 - Alerts & Notifications",
            "08 - Performance Trends & Analytics",
            "09 - Knowledge Store & History",
            "10 - Settings & Configuration",
            "11 - Connector Management",
            "12 - E-Commerce Application",
            "13 - Reports & Export",
            "14 - About & Help",
            "Logout"
        ]

    page = st.sidebar.radio(
        "MAIN NAVIGATION",
        navigation,
        index=0
    )

    st.sidebar.divider()

    st.sidebar.markdown("### Logged-in User")
    st.sidebar.info(
        f"**Role:** {st.session_state.user_role}\n\n"
        f"**Email:** {st.session_state.current_user}"
    )

    st.sidebar.markdown("### System Status")
    st.sidebar.success("Demo data source connected")
    st.sidebar.success("ML engine ready")
    st.sidebar.warning("Ollama SLM optional")

    return page



# ---------- Admin Panel ----------

# ---------- Admin Panel ----------

def admin_panel():

    st.title("Admin Control Center")
    st.caption("TwinOps System Administration and Monitoring")

    st.divider()

    st.subheader("System Overview")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Registered Users", "2")
    c2.metric("Active Roles", "2")
    c3.metric("Application Status", "Active")
    c4.metric("ML Engine", "Ready")

    st.divider()


    # User Management
    st.subheader("User Management")

    users = pd.DataFrame({
        "User ID": ["USR001", "USR002"],
        "Role": ["Admin", "Developer"],
        "Email": [
            "admin@twinops.com",
            "developer@twinops.com"
        ],
        "Access Level": [
            "Full Administrative Access",
            "Developer Access"
        ],
        "Status": ["Active", "Active"]
    })

    st.dataframe(
        users,
        hide_index=True,
        use_container_width=True
    )

    st.divider()

    # System Health
    
    st.subheader("System Health")

    h1, h2, h3 = st.columns(3)

    h1.metric("CPU Utilization", "43%")
    h2.metric("Memory Utilization", "55%")
    h3.metric("Average Response", "900 ms")

    st.caption(
        "Illustrative demo metrics. "
        "These values are not collected from a live production server."
    )

    st.divider()


    # User Access Management
    st.subheader("User Access Management")

    selected_user = st.selectbox(
        "Select User",
        [
            "admin@twinops.com",
            "developer@twinops.com"
        ]
    )

    if selected_user == "admin@twinops.com":
        st.write("**Role:** Admin")
        st.success("Full administrative access")
        st.write("Permissions: Dashboard, Admin Panel, ML, Simulation")

    else:
        st.write("**Role:** Developer")
        st.info("Developer access")
        st.write("Permissions: Dashboard, Digital Twin, ML, Simulation")

    st.caption(
        "This is a permission overview. It does not modify real account credentials."
    )

    st.divider()


    # AI and SLM status
    st.subheader("AI & SLM Services")

    a, b = st.columns(2)

    with a:
        st.markdown("**Machine Learning Engine**")
        st.success("Isolation Forest: Ready")
        st.success("Random Forest: Ready")

    with b:
        st.markdown("**Small Language Model**")
        st.info("Ollama: Optional local service")
        st.caption("SLM availability depends on local Ollama setup.")

    st.divider()

    # Activity overview
    st.subheader("System Activity")

    activity = pd.DataFrame({
        "Activity": [
            "Admin account configured",
            "Developer account configured",
            "Dashboard module available",
            "ML analysis module available",
            "Simulation module available"
        ],
        "Status": [
            "Completed",
            "Completed",
            "Available",
            "Available",
            "Available"
        ]
    })

    st.dataframe(
        activity,
        hide_index=True,
        use_container_width=True
    )

    st.info(
        "TwinOps Admin Panel uses demonstration account and sample system data. "
        "It does not manage real users or monitor a production application."
    )


# ---------- Dashboard ----------
def dashboard(history, current):
    st.title("TwinOps Overview")
    st.caption("Monitor the sample application and review its current performance.")
    health = max(0, min(100, 100 - max(0,current['cpu']-60)*0.55 - max(0,current['response_time']-1)*12 - current['error_rate']*2))
    
    st.title("TwinOps Dashboard")
    st.caption("Software Performance Monitoring and Intelligence")

    st.divider()

    st.subheader("Live System Overview")

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric(
        "CPU Usage",
        f"{current['cpu']}%",
        "System utilization"
    )

    c2.metric(
        "Memory Usage",
        f"{current['memory']}%",
        "Memory utilization"
    )

    c3.metric(
        "Incoming Requests",
        f"{int(current['requests'])}",
        "Request count"
    )

    c4.metric(
        "Response Time",
        f"{current['response_time']} s",
        "Average response"
    )

    c5.metric(
        "Error Rate",
        f"{current['error_rate']}%",
        "Application errors"
    )

    st.divider()

    st.subheader("Performance Analytics")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### CPU Usage Trend")

        fig_cpu = px.line(
            history,
            x=history.index,
            y="cpu",
            title="CPU Utilization"
        )

        st.plotly_chart(fig_cpu, use_container_width=True)

    with col2:
        st.markdown("#### Memory Usage Trend")

        fig_memory = px.line(
            history,
            x=history.index,
            y="memory",
            title="Memory Utilization"
        )

        st.plotly_chart(fig_memory, use_container_width=True)

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

# ---------- Main ----------

# ---------- Main App ----------

history = make_history()

if not st.session_state.logged_in:
    login_page()

else:
    page = sidebar()

    current = {
        "cpu": 43.0,
        "memory": 55.0,
        "requests": 620.0,
        "response_time": 0.9,
        "error_rate": 1.0
    }

    if page == "Dashboard":
        dashboard(history, current)

    elif page == "Admin Panel":
        if st.session_state.user_role == "Admin":
            admin_panel()
        else:
            st.error("Access denied. Admin permission required.")

    elif page == "Digital Twin Topology":
        topology(current)

    elif page == "ML & SLM Analysis":
        ml_slm_page(history)

    elif page == "Simulation & Trust":
        simulation_page()

    elif page == "Logout":
        st.session_state.logged_in = False
        st.session_state.user_role = None
        st.session_state.current_user = None
        st.rerun()

    else:
        st.title(page)
        st.info("This section is reserved for the next development stage.")