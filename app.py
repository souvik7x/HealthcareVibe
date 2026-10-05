"""
app.py — HealthCare Analytics dashboard (Streamlit)

Run with:
    streamlit run app.py
"""

import base64
import time
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from streamlit.components.v1 import html as components_html

from utils import LABELS, N_RECORDS, bmi_category, generate_dataset, predict_risk, train_model

LOGO_PATH = Path(__file__).parent / "assets" / "logo.png"
FAVICON_PATH = Path(__file__).parent / "assets" / "favicon.png"


@st.cache_data
def get_logo_base64() -> str:
    return base64.b64encode(LOGO_PATH.read_bytes()).decode()


st.set_page_config(
    page_title="HealthCare Analytics",
    page_icon=str(FAVICON_PATH) if FAVICON_PATH.exists() else "💙",
    layout="wide",
)

# ------------------------------------------------------------------ login
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = "Admin"


def login_screen():
    st.markdown(
        """
        <style>
        [data-testid="stAppViewContainer"] {
            background:
                radial-gradient(circle at 12% 8%, rgba(59,130,246,0.25) 0%, transparent 38%),
                radial-gradient(circle at 88% 15%, rgba(34,211,238,0.18) 0%, transparent 35%),
                radial-gradient(circle at 50% 95%, rgba(168,85,247,0.16) 0%, transparent 40%),
                #0a0f1d;
        }
        [data-testid="stHeader"] { background: transparent; }

        @keyframes hc-pulse {
            0%   { box-shadow: 0 0 0 0 rgba(59,130,246,0.45), 0 10px 30px rgba(59,130,246,0.35); }
            70%  { box-shadow: 0 0 0 14px rgba(59,130,246,0), 0 10px 30px rgba(59,130,246,0.35); }
            100% { box-shadow: 0 0 0 0 rgba(59,130,246,0), 0 10px 30px rgba(59,130,246,0.35); }
        }
        .login-logo {
            width:58px; height:58px; border-radius:17px;
            overflow:hidden; margin: 6px auto 14px auto;
            animation: hc-pulse 2.6s ease-out infinite;
        }
        .login-logo img { width:100%; height:100%; object-fit:cover; display:block; }
        .login-title {
            text-align:center; color:white; font-size:23px; font-weight:800; margin-bottom:2px;
            letter-spacing:.2px;
        }
        .login-tagline {
            text-align:center; color:#64748b; font-size:11.5px; margin-bottom:20px;
            letter-spacing:.4px;
        }
        .login-tagline b { color:#22d3ee; font-weight:600; }

        div[data-testid="stVerticalBlockBorderWrapper"]:has(div.login-marker) {
            background: rgba(255,255,255,0.045) !important;
            backdrop-filter: blur(14px);
            border: 1px solid rgba(255,255,255,0.09) !important;
            border-radius: 20px !important;
            box-shadow: 0 20px 60px rgba(0,0,0,0.5);
        }

        .login-field-label {
            color:#94a3b8; font-size:12px; font-weight:600; margin: 2px 0 2px 2px;
            letter-spacing:.3px;
        }
        div[data-testid="stTextInput"] input {
            background: rgba(255,255,255,0.045) !important;
            border: 1px solid rgba(255,255,255,0.12) !important;
            border-radius: 10px !important; color: #e2e8f0 !important;
        }
        div[data-testid="stTextInput"] input:focus {
            border-color: #3b82f6 !important;
            box-shadow: 0 0 0 3px rgba(59,130,246,0.22) !important;
        }
        div[data-testid="stFormSubmitButton"] button {
            background: linear-gradient(90deg,#3b82f6,#2563eb) !important;
            color: white !important; border: none !important;
            border-radius: 10px !important; font-weight: 600 !important;
            padding: 10px 0 !important; margin-top: 10px;
            transition: transform .12s, box-shadow .12s;
        }
        div[data-testid="stFormSubmitButton"] button:hover {
            box-shadow: 0 6px 20px rgba(59,130,246,0.45);
            transform: translateY(-1px);
        }

        [data-testid="stButton"] button {
            background: rgba(255,255,255,0.04) !important;
            border: 1px solid rgba(255,255,255,0.14) !important;
            color: #e2e8f0 !important; border-radius: 10px !important;
            font-weight: 600 !important; padding: 9px 0 !important;
        }
        [data-testid="stButton"] button:hover {
            background: rgba(255,255,255,0.08) !important;
            border-color: rgba(255,255,255,0.25) !important;
        }

        .login-divider {
            display:flex; align-items:center; gap:10px; margin: 16px 0 12px 0;
            color:#475569; font-size:11px; letter-spacing:.3px;
        }
        .login-divider::before, .login-divider::after {
            content:""; flex:1; height:1px; background: rgba(255,255,255,0.1);
        }
        .login-foot { text-align:center; color:#475569; font-size:11px; margin-top:16px; }
        .login-page-footer { text-align:center; color:#334155; font-size:11.5px; margin-top:22px; }

        /* Pin the particle-background iframe full-screen, behind everything */
        iframe { position: fixed !important; top:0; left:0; width:100% !important;
                 height:100% !important; z-index:0; pointer-events:none; border:none; }
        [data-testid="stAppViewContainer"] > .main { position: relative; z-index: 1; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Floating particle background + Caps Lock detector (runs once, in a
    # sandboxed component iframe that reaches into the parent page's DOM).
    components_html(
        """
        <script>
        (function () {
            if (window.parent.document.getElementById('hc-particles')) { return; }

            // ---- floating particles ----
            const canvas = document.createElement('canvas');
            canvas.id = 'hc-particles';
            canvas.style.position = 'fixed';
            canvas.style.top = '0'; canvas.style.left = '0';
            canvas.style.width = '100%'; canvas.style.height = '100%';
            canvas.style.zIndex = '0'; canvas.style.pointerEvents = 'none';
            window.parent.document.body.appendChild(canvas);

            const ctx = canvas.getContext('2d');
            function resize() {
                canvas.width = window.parent.innerWidth;
                canvas.height = window.parent.innerHeight;
            }
            resize();
            window.parent.addEventListener('resize', resize);

            const colors = ['rgba(59,130,246,', 'rgba(34,211,238,', 'rgba(168,85,247,'];
            const particles = Array.from({ length: 45 }, () => ({
                x: Math.random() * canvas.width,
                y: Math.random() * canvas.height,
                r: Math.random() * 2.6 + 1,
                dx: (Math.random() - 0.5) * 0.25,
                dy: (Math.random() - 0.5) * 0.25,
                o: Math.random() * 0.35 + 0.08,
                c: colors[Math.floor(Math.random() * colors.length)],
            }));

            function animate() {
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                particles.forEach((p) => {
                    p.x += p.dx; p.y += p.dy;
                    if (p.x < 0) p.x = canvas.width;
                    if (p.x > canvas.width) p.x = 0;
                    if (p.y < 0) p.y = canvas.height;
                    if (p.y > canvas.height) p.y = 0;
                    ctx.beginPath();
                    ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
                    ctx.fillStyle = p.c + p.o + ')';
                    ctx.fill();
                });
                requestAnimationFrame(animate);
            }
            animate();

            // ---- Caps Lock warning on the password field ----
            function attachCapsLock() {
                const pwInput = window.parent.document.querySelector('input[type="password"]');
                const warn = window.parent.document.getElementById('capslock-warning');
                if (!pwInput || !warn) { setTimeout(attachCapsLock, 300); return; }
                pwInput.addEventListener('keyup', function (e) {
                    const isCaps = e.getModifierState && e.getModifierState('CapsLock');
                    warn.style.display = isCaps ? 'block' : 'none';
                });
            }
            attachCapsLock();
        })();
        </script>
        """,
        height=0,
    )

    st.markdown("<div style='height:48px'></div>", unsafe_allow_html=True)
    _, col, _ = st.columns([1, 1.05, 1])
    with col:
        with st.container(border=True):
            st.markdown('<div class="login-marker"></div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="login-logo"><img src="data:image/png;base64,{get_logo_base64()}"></div>',
                unsafe_allow_html=True,
            )
            st.markdown('<div class="login-title">HealthCare Analytics</div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="login-tagline">Admin Panel &nbsp;&bull;&nbsp; '
                '<b>Smarter Insights, Better Decisions</b></div>',
                unsafe_allow_html=True,
            )

            with st.form("login_form"):
                st.markdown('<div class="login-field-label">👤 Username</div>', unsafe_allow_html=True)
                user = st.text_input("Username", value="admin", label_visibility="collapsed")
                st.markdown('<div class="login-field-label">🔒 Password</div>', unsafe_allow_html=True)
                pw = st.text_input("Password", type="password", label_visibility="collapsed")
                st.markdown(
                    '<div id="capslock-warning" style="display:none; color:#f87171; '
                    'font-size:11.5px; margin:-6px 2px 8px 2px;">⚠️ Caps Lock is ON</div>',
                    unsafe_allow_html=True,
                )
                ok = st.form_submit_button("→  Sign in", use_container_width=True)

            st.markdown('<div class="login-divider">or continue with</div>', unsafe_allow_html=True)
            google_clicked = st.button("🔵  Continue with Google", use_container_width=True)

            st.markdown('<div class="login-foot">Demo password: admin123</div>', unsafe_allow_html=True)

    st.markdown(
        '<div class="login-page-footer">© 2026 HealthCare Analytics &nbsp;|&nbsp; '
        'Secure &bull; Trusted &bull; For a Healthier Tomorrow</div>',
        unsafe_allow_html=True,
    )

    if ok:
        with st.spinner("Signing in..."):
            time.sleep(0.7)
        if user.strip() and pw == "admin123":
            st.session_state.logged_in = True
            st.session_state.username = user.strip().title()
            st.rerun()
        else:
            st.error("Incorrect username or password. (demo password: admin123)")

    if google_clicked:
        try:
            st.login("google")
        except Exception:
            st.warning(
                "Google sign-in isn't configured yet. Add your Google OAuth "
                "credentials to `.streamlit/secrets.toml` (see README) to enable this button."
            )


if hasattr(st, "user") and getattr(st.user, "is_logged_in", False):
    st.session_state.logged_in = True
    st.session_state.username = (st.user.name or st.user.email or "Google User").split("@")[0].title()
    st.session_state.role = "Google Account"

if not st.session_state.logged_in:
    login_screen()
    st.stop()

# ------------------------------------------------------------- app state
THEMES = {"Blue": "#3b82f6", "Green": "#22c55e", "Purple": "#a855f7", "Pink": "#ec4899"}
if "theme_color" not in st.session_state:
    st.session_state.theme_color = "Blue"
if "n_records" not in st.session_state:
    st.session_state.n_records = N_RECORDS
accent = THEMES[st.session_state.theme_color]

# --------------------------------------------------------------- data/model
@st.cache_data(show_spinner="Generating patient dataset...")
def load_data(n_records):
    return generate_dataset(n_records=n_records)


@st.cache_resource(show_spinner="Training risk model...")
def load_model(df):
    return train_model(df)


df = load_data(st.session_state.n_records)
bundle = load_model(df)

# --------------------------------------------------------------------- CSS
st.markdown(
    """
    <style>
    [data-testid="stMetricValue"] { font-size: 1.6rem; }
    .block-container { padding-top: 1.5rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------- sidebar
if "page" not in st.session_state:
    st.session_state.page = "Dashboard"

NAV_GROUPS = {
    "MAIN": [
        ("🏠", "Dashboard"),
        ("📈", "Healthcare Analytics"),
        ("📉", "Risk Factor Analysis"),
        ("🎯", "Risk Prediction"),
        ("🕒", "Model Performance"),
    ],
    "MANAGE": [
        ("🗂️", "Patient Records"),
        ("⚙️", "Settings"),
    ],
    "ACCOUNT": [
        ("🚪", "Logout"),
    ],
}

with st.sidebar:
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0b1220 0%, #111827 100%);
            border-right: 1px solid rgba(255,255,255,0.05);
        }
        .sidebar-logo-box { display:flex; align-items:center; gap:10px; margin-bottom:2px; }
        .sidebar-logo-icon {
            width:36px; height:36px; border-radius:10px; overflow:hidden;
        }
        .sidebar-logo-icon img { width:100%; height:100%; object-fit:cover; display:block; }
        .sidebar-logo-text { font-size:17px; font-weight:800; color:white; line-height:1.15; }
        .sidebar-logo-text span { color:#22d3ee; display:block; font-size:11px; font-weight:700; letter-spacing:1.5px; }

        .sidebar-avatar {
            display:flex; align-items:center; gap:10px;
            background: rgba(255,255,255,0.04);
            border: 1px solid rgba(255,255,255,0.06);
            padding: 10px 12px; border-radius: 12px; margin: 18px 0 8px 0;
        }
        .sidebar-avatar .circle {
            width:38px; height:38px; border-radius:50%;
            background:#334155; display:flex; align-items:center; justify-content:center;
            font-size:17px; position:relative; flex-shrink:0;
        }
        .sidebar-avatar .dot {
            width:9px; height:9px; border-radius:50%; background:#22c55e;
            position:absolute; bottom:-1px; right:-1px; border:2px solid #111827;
        }
        .sidebar-avatar .name { color:white; font-weight:700; font-size:13.5px; line-height:1.3; }
        .sidebar-avatar .role { color:#94a3b8; font-size:11px; }

        .nav-section-label {
            color:#64748b; font-size:10.5px; font-weight:700; letter-spacing:1.3px;
            text-transform:uppercase; margin: 16px 4px 4px 4px;
        }

        div[data-baseweb="radio"] > div:first-child { display: none; }
        div[role="radiogroup"] { gap: 2px; }
        div[role="radiogroup"] > label {
            padding: 8px 12px; border-radius: 8px; margin-bottom: 2px;
            transition: background 0.15s; color:#cbd5e1 !important;
        }
        div[role="radiogroup"] > label:hover { background-color: rgba(59,130,246,0.12); }
        div[role="radiogroup"] > label:has(input:checked) {
            background: linear-gradient(90deg, #3b82f6, #2563eb);
            box-shadow: 0 2px 10px rgba(59,130,246,0.35);
        }
        div[role="radiogroup"] > label:has(input:checked) p {
            color:white !important; font-weight:600;
        }

        .sidebar-footer {
            margin-top: 26px; padding-top:12px; border-top:1px solid rgba(255,255,255,0.06);
            color:#475569; font-size:10.5px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <div class="sidebar-logo-box">
            <div class="sidebar-logo-icon"><img src="data:image/png;base64,{get_logo_base64()}"></div>
            <div class="sidebar-logo-text">HealthCare<span>ANALYTICS</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="sidebar-avatar">
            <div class="circle">🧑<div class="dot"></div></div>
            <div>
                <div class="name">{st.session_state.username}</div>
                <div class="role">Administrator</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    label_to_page = {
        f"{icon}  {name}": name for items in NAV_GROUPS.values() for icon, name in items
    }

    for section, items in NAV_GROUPS.items():
        st.markdown(f'<div class="nav-section-label">{section}</div>', unsafe_allow_html=True)
        options = [f"{icon}  {name}" for icon, name in items]
        current = next((opt for opt in options if label_to_page[opt] == st.session_state.page), None)
        idx = options.index(current) if current else None
        choice = st.radio(
            section, options, index=idx, key=f"nav_{section}", label_visibility="collapsed",
        )
        if choice and label_to_page[choice] != st.session_state.page:
            st.session_state.page = label_to_page[choice]
            # Reset the other groups' widget state so they don't keep
            # showing their old selection on the next run.
            for other_section in NAV_GROUPS:
                if other_section != section:
                    st.session_state.pop(f"nav_{other_section}", None)
            st.rerun()

    st.markdown(
        f"""
        <div class="sidebar-footer">{len(df):,} records &bull; v1.0</div>
        """,
        unsafe_allow_html=True,
    )

page = st.session_state.page

# ========================================================== DASHBOARD PAGE
if page == "Dashboard":
    st.title("Dashboard Overview")
    st.caption("Healthcare Analytics & Pre-diabetes Risk Prediction")

    total = len(df)
    counts = df["Diagnosis"].value_counts()
    healthy, pre, dia = (counts.get(label, 0) for label in LABELS)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Records", f"{total:,}", "100% of data")
    c2.metric("Healthy", f"{healthy:,}", f"{healthy/total:.2%}")
    c3.metric("Pre-diabetes", f"{pre:,}", f"{pre/total:.2%}")
    c4.metric("Diabetes", f"{dia:,}", f"{dia/total:.2%}")
    overall = "Moderate" if pre / total > 0.15 else "Low"
    c5.metric("Overall Risk", overall, "Keep monitoring")

    st.markdown("###")
    col1, col2 = st.columns([1, 1.4])

    with col1:
        st.subheader("Risk Distribution")
        fig = go.Figure(
            go.Pie(
                labels=counts.index,
                values=counts.values,
                hole=0.55,
                marker_colors=["#22c55e", "#f59e0b", "#ef4444"],
            )
        )
        fig.update_layout(margin=dict(t=0, b=0, l=0, r=0), height=320)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Age Distribution")
        bins = [0, 20, 30, 40, 50, 60, 90]
        age_labels = ["0-20", "21-30", "31-40", "41-50", "51-60", "60+"]
        age_grp = pd.cut(df["Age"], bins=bins, labels=age_labels, right=True)
        age_counts = age_grp.value_counts().reindex(age_labels)
        fig = px.bar(
            x=age_counts.index, y=age_counts.values,
            labels={"x": "Age Group", "y": "Count"},
        )
        fig.update_traces(marker_color=accent)
        fig.update_layout(margin=dict(t=0, b=0, l=0, r=0), height=320)
        st.plotly_chart(fig, use_container_width=True)

    col3, col4, col5 = st.columns(3)
    with col3:
        st.subheader("BMI Distribution")
        bmi_cat = bmi_category(df["BMI"]).value_counts()
        fig = go.Figure(
            go.Pie(
                labels=bmi_cat.index, values=bmi_cat.values,
                marker_colors=["#3b82f6", "#22c55e", "#f59e0b", "#ef4444"],
            )
        )
        fig.update_layout(margin=dict(t=0, b=0, l=0, r=0), height=280)
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        st.subheader("Blood Glucose Distribution")
        fig = px.histogram(df, x="BloodGlucose", nbins=20)
        fig.update_traces(marker_color="#a855f7")
        fig.update_layout(margin=dict(t=0, b=0, l=0, r=0), height=280, bargap=0.1)
        st.plotly_chart(fig, use_container_width=True)

    with col5:
        st.subheader("Gender Distribution")
        gcounts = df["Gender"].value_counts()
        fig = go.Figure(
            go.Pie(
                labels=gcounts.index, values=gcounts.values, hole=0.55,
                marker_colors=["#ec4899", "#3b82f6"],
            )
        )
        fig.update_layout(margin=dict(t=0, b=0, l=0, r=0), height=280)
        st.plotly_chart(fig, use_container_width=True)

# =================================================== HEALTHCARE ANALYTICS
elif page == "Healthcare Analytics":
    st.title("Healthcare Analytics")
    st.caption("Deeper cuts of the patient population")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Average glucose by age group")
        bins = [0, 20, 30, 40, 50, 60, 90]
        age_labels = ["0-20", "21-30", "31-40", "41-50", "51-60", "60+"]
        tmp = df.copy()
        tmp["AgeGroup"] = pd.cut(tmp["Age"], bins=bins, labels=age_labels)
        avg_glucose = tmp.groupby("AgeGroup", observed=True)["BloodGlucose"].mean().reindex(age_labels)
        fig = px.line(
            x=avg_glucose.index, y=avg_glucose.values, markers=True,
            labels={"x": "Age Group", "y": "Avg Blood Glucose"},
        )
        fig.update_traces(line_color=accent)
        fig.update_layout(height=340, margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("BMI vs Blood Glucose")
        sample = df.sample(min(3000, len(df)), random_state=1)
        fig = px.scatter(
            sample, x="BMI", y="BloodGlucose", color="Diagnosis",
            category_orders={"Diagnosis": LABELS},
            color_discrete_map={"Healthy": "#22c55e", "Pre-diabetes": "#f59e0b", "Diabetes": "#ef4444"},
            opacity=0.5,
        )
        fig.update_layout(height=340, margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("At-risk rate (%) by activity level and gender")
    pivot = (
        df.assign(AtRisk=df["Diagnosis"] != "Healthy")
        .pivot_table(index="PhysicalActivity", columns="Gender", values="AtRisk", aggfunc="mean")
        .reindex(["Low", "Moderate", "High"])
        * 100
    )
    fig = px.imshow(
        pivot, text_auto=".1f", color_continuous_scale="Oranges",
        labels=dict(color="At-risk %"),
    )
    fig.update_layout(height=320, margin=dict(t=10))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Summary statistics by diagnosis")
    summary = (
        df.groupby("Diagnosis")[["Age", "BMI", "BloodGlucose", "Cholesterol", "HbA1c"]]
        .mean()
        .round(1)
        .reindex(LABELS)
    )
    st.dataframe(summary, use_container_width=True)

# =================================================== RISK FACTOR ANALYSIS
elif page == "Risk Factor Analysis":
    st.title("Risk Factor Analysis")
    st.caption("How each factor relates to Pre-diabetes / Diabetes risk")

    categorical_factors = ["Gender", "PhysicalActivity", "FamilyHistory", "Smoking"]
    numeric_factors = ["Age", "BMI", "BloodGlucose", "SystolicBP", "Cholesterol", "HbA1c"]

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("At-risk rate by category")
        factor = st.selectbox("Choose a factor", categorical_factors)
        rate = (
            df.assign(AtRisk=df["Diagnosis"] != "Healthy")
            .groupby(factor)["AtRisk"]
            .mean()
            .sort_values(ascending=False)
        )
        fig = px.bar(
            x=rate.index.astype(str), y=rate.values * 100,
            labels={"x": factor, "y": "At-risk rate (%)"},
        )
        fig.update_traces(marker_color="#ef4444")
        fig.update_layout(height=350, margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Distribution by diagnosis")
        num_factor = st.selectbox("Choose a measurement", numeric_factors)
        fig = px.box(
            df, x="Diagnosis", y=num_factor, color="Diagnosis",
            category_orders={"Diagnosis": LABELS},
            color_discrete_map={"Healthy": "#22c55e", "Pre-diabetes": "#f59e0b", "Diabetes": "#ef4444"},
        )
        fig.update_layout(height=350, margin=dict(t=10), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Feature correlation")
    corr = df[numeric_factors].corr()
    fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
    fig.update_layout(height=420, margin=dict(t=10))
    st.plotly_chart(fig, use_container_width=True)

# ==================================================== RISK PREDICTION PAGE
elif page == "Risk Prediction":
    st.title("Quick Risk Prediction")

    with st.form("risk_form"):
        c1, c2 = st.columns(2)
        with c1:
            age = st.number_input("Age", 1, 100, 45)
            bmi = st.number_input("BMI", 10.0, 60.0, 28.5, step=0.1)
            glucose = st.number_input("Blood Glucose (mg/dL)", 50, 400, 145)
            systolic = st.number_input("Systolic BP", 80, 220, 130)
        with c2:
            activity = st.selectbox("Physical Activity", ["Low", "Moderate", "High"], index=1)
            family = st.selectbox("Family History", ["Yes", "No"], index=0)
            cholesterol = st.number_input("Cholesterol", 100, 400, 190)
            smoking = st.selectbox("Smoking", ["Yes", "No"], index=1)
        submitted = st.form_submit_button("Predict Risk", use_container_width=True)

    if submitted:
        pred, prob, all_proba = predict_risk(
            bundle, age, bmi, glucose, systolic, activity, family,
            cholesterol=cholesterol, smoking=smoking,
        )
        color = {"Healthy": "#22c55e", "Pre-diabetes": "#f59e0b", "Diabetes": "#ef4444"}[pred]
        st.markdown(
            f"### Predicted Risk Category: "
            f"<span style='color:{color}'>{pred}</span>",
            unsafe_allow_html=True,
        )
        st.markdown(f"**Risk Probability:** {prob:.2%}")

        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": color},
                    "steps": [
                        {"range": [0, 33], "color": "#dcfce7"},
                        {"range": [33, 66], "color": "#fef3c7"},
                        {"range": [66, 100], "color": "#fee2e2"},
                    ],
                },
            )
        )
        fig.update_layout(height=300, margin=dict(t=20, b=0))
        st.plotly_chart(fig, use_container_width=True)

        st.bar_chart(pd.Series(all_proba))

# ================================================== MODEL PERFORMANCE PAGE
elif page == "Model Performance":
    st.title("Model Performance")
    m = bundle["metrics"]
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Accuracy", f"{m['accuracy']:.2%}")
    c2.metric("Precision", f"{m['precision']:.2%}")
    c3.metric("Recall", f"{m['recall']:.2%}")
    c4.metric("F1-Score", f"{m['f1']:.2%}")
    c5.metric("ROC-AUC", f"{m['roc_auc']:.2f}")

    col1, col2 = st.columns([1, 1.3])
    with col1:
        st.subheader("Confusion Matrix")
        cm = bundle["confusion_matrix"]
        fig = px.imshow(
            cm, x=LABELS, y=LABELS, text_auto=True, color_continuous_scale="Blues",
            labels=dict(x="Predicted", y="Actual", color="Count"),
        )
        fig.update_layout(height=380)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Top Important Features")
        imp = bundle["importances"]
        fig = px.bar(
            x=imp.values, y=imp.index, orientation="h",
            labels={"x": "Importance Score", "y": ""},
        )
        fig.update_traces(marker_color="#22c55e")
        fig.update_layout(height=380, yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)

# ======================================================= PATIENT RECORDS
elif page == "Patient Records":
    st.title("Patient Records")

    records = df.copy()
    records.insert(0, "PatientID", [f"P{100000 + i}" for i in range(len(records))])

    c1, c2, c3 = st.columns(3)
    with c1:
        diag_filter = st.multiselect("Diagnosis", LABELS, default=LABELS)
    with c2:
        genders = records["Gender"].unique().tolist()
        gender_filter = st.multiselect("Gender", genders, default=genders)
    with c3:
        age_min, age_max = int(records["Age"].min()), int(records["Age"].max())
        age_range = st.slider("Age range", age_min, age_max, (20, 60))

    filtered = records[
        records["Diagnosis"].isin(diag_filter)
        & records["Gender"].isin(gender_filter)
        & records["Age"].between(*age_range)
    ]

    st.caption(f"{len(filtered):,} of {len(records):,} records match your filters")
    st.dataframe(filtered.head(2000), use_container_width=True, height=460)
    if len(filtered) > 2000:
        st.caption("Showing the first 2,000 matching rows. Download the CSV for the full filtered set.")

    csv = filtered.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download filtered records (CSV)", csv, "patient_records.csv", "text/csv",
        use_container_width=True,
    )

# =============================================================== SETTINGS
elif page == "Settings":
    st.title("Settings")

    st.subheader("Dataset")
    new_n = st.number_input(
        "Number of synthetic records", min_value=1000, max_value=300000,
        value=st.session_state.n_records, step=1000,
    )
    if st.button("Regenerate dataset"):
        st.session_state.n_records = int(new_n)
        load_data.clear()
        load_model.clear()
        st.success("Dataset regenerated.")
        st.rerun()

    st.subheader("Model")
    st.write("Algorithm: RandomForestClassifier (300 trees, max depth 12)")
    if st.button("Retrain model"):
        load_model.clear()
        st.success("Model retrained.")
        st.rerun()

    st.subheader("Appearance")
    theme_choice = st.selectbox(
        "Chart accent color", list(THEMES.keys()),
        index=list(THEMES.keys()).index(st.session_state.theme_color),
    )
    if theme_choice != st.session_state.theme_color:
        st.session_state.theme_color = theme_choice
        st.rerun()

    st.subheader("Account")
    st.write(f"Signed in as **{st.session_state.username}**")

# ================================================================= LOGOUT
elif page == "Logout":
    st.title("Logout")
    st.write(f"You are signed in as **{st.session_state.username}**.")
    if st.button("Confirm logout", type="primary"):
        if hasattr(st, "user") and getattr(st.user, "is_logged_in", False):
            st.logout()
        st.session_state.logged_in = False
        st.rerun()