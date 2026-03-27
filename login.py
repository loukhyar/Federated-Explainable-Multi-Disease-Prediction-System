import streamlit as st
import pandas as pd

st.set_page_config(page_title="Auth", layout="centered")

# =========================
# STYLE
# =========================
st.markdown("""
<style>
.stApp{
background:linear-gradient(135deg,#f6f0ff,#ffffff);
font-family:Segoe UI;
}

h1{
text-align:center;
color:#5e4aa8;
}

div.stButton > button{
background:linear-gradient(135deg,#8b7af7,#6c5ce7);
color:white;
border-radius:10px;
padding:10px 20px;
border:none;
transition:0.3s;
}

div.stButton > button:hover{
transform:scale(1.05);
}

/* Sidebar */
[data-testid="stSidebar"]{
background:linear-gradient(180deg,#efe7ff,#ffffff);
border-right:1px solid #e6dbff;
}

[data-testid="stSidebar"] *{
color:#4b3fa3;
}

.footer{
text-align:center;
color:#777;
font-size:13px;
margin-top:30px;
}
</style>
""", unsafe_allow_html=True)

# =========================
# SESSION
# =========================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# =========================
# AUTO REDIRECT
# =========================
if st.session_state.logged_in:
    st.switch_page("pages/App.py")

# =========================
# SIDEBAR (UPGRADED 💜)
# =========================
with st.sidebar:

    st.markdown("## 🩺 Health AI Panel")
    st.write("✨ Smart Healthcare Assistant")

    st.markdown("---")

    st.info("🔐 Secure login required")

    st.markdown("### 🌟 Features")

    st.markdown("""
    💜 Multi-Disease Prediction  
    🔬 Explainable AI (SHAP + LIME)  
    📊 Risk Analysis  
    🏥 Medical Insights  
    """)

    st.markdown("---")

    st.success("🟢 System Active")

    st.markdown("### 💡 Tip")
    st.write("Enter accurate data for better predictions ✨")

# =========================
# TITLE
# =========================
st.markdown("<h1>🔐 Welcome to Healthcare AI</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:#666'>Your AI-powered medical assistant 💜</p>", unsafe_allow_html=True)

# =========================
# TABS
# =========================
tab1, tab2 = st.tabs(["Login", "Sign Up"])

# =========================
# LOGIN
# =========================
with tab1:

    st.subheader("Login")

    username = st.text_input("Username", key="login_user")
    password = st.text_input("Password", type="password", key="login_pass")

    if st.button("Login", use_container_width=True):

        try:
            users = pd.read_csv("users.csv")
        except:
            st.error("users.csv not found")
            st.stop()

        user = users[
            (users["username"] == username) &
            (users["password"] == password)
        ]

        if not user.empty:
            st.session_state.logged_in = True
            st.session_state.user = username

            st.success("Login successful ✅")
            st.switch_page("pages/App.py")
        else:
            st.error("Invalid credentials")

    st.markdown(
    "<p style='text-align:center;color:#777;font-size:14px;'>✨ Your health, powered by AI</p>",
    unsafe_allow_html=True
    )

# =========================
# SIGNUP
# =========================
with tab2:

    st.subheader("Create Account")

    new_user = st.text_input("Username", key="signup_user")
    new_pass = st.text_input("Password", type="password", key="signup_pass")

    if st.button("✨ Sign Up", use_container_width=True):

        try:
            users = pd.read_csv("users.csv")
        except:
            users = pd.DataFrame(columns=["username", "password"])

        if new_user in users["username"].values:
            st.warning("User already exists")
        else:
            new_data = pd.DataFrame([[new_user, new_pass]], columns=["username", "password"])
            users = pd.concat([users, new_data], ignore_index=True)
            users.to_csv("users.csv", index=False)

            st.success("Account created successfully 🎉")
            st.info("Switch to Login tab and continue")

# =========================
# FOOTER
# =========================
st.markdown("<div class='footer'>Federated Healthcare AI • Secure Medical Platform 💜</div>", unsafe_allow_html=True)