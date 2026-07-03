import os
import base64
import re
import random
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report

# ==============================================================================
# 1. PERSISTENT USER REGISTRY ENGINE WITH EXTENDED DATA SCHEMA
# ==============================================================================
USER_DB_FILE = "users_db.csv"

def init_user_db():
    if not os.path.exists(USER_DB_FILE):
        columns = [
            "Username",
            "Password",
            "Secret_Question",
            "Secret_Answer",
            "First_Name",
            "Last_Name",
            "Phone",
            "Email",
            "Registration_Date",
            "Last_Login",
            "Last_Logout"
        ]
        df_users = pd.DataFrame(columns=columns)
        
        # Seed default administrative profile (Fixed uppercase 'S' to lowercase 's')
        admin_row = pd.DataFrame([{
            "Username": "admin",
            "Password": "password123",
            "Secret_Question": "What is your favourite colour?",
            "Secret_Answer": "green",
            "First_Name": "System",
            "Last_Name": "Admin",
            "Phone": "9876543210",
            "Email": "admin@facility.com",
            "Registration_Date": str(pd.Timestamp.now().round('s')),
            "Last_Login": "Never",
            "Last_Logout": "Never"
        }])
        df_users = pd.concat([df_users, admin_row], ignore_index=True)
        df_users.to_csv(USER_DB_FILE, index=False)

def get_users_df():
    init_user_db()
    df = pd.read_csv(USER_DB_FILE)
   
    required_cols = [
        "Username", "Password", "First_Name", "Last_Name", 
        "Phone", "Email", "Registration_Date", "Last_Login", 
        "Last_Logout","Secret_Question","Secret_Answer",
    ]
    is_updated = False
    for col in required_cols:
        if col not in df.columns:
            df[col] = "N/A"
            is_updated = True
    df["Phone"] = df["Phone"].astype(str)
    if is_updated:
        df.to_csv(USER_DB_FILE, index=False)
    return df

def save_users_df(df):
    df.to_csv(USER_DB_FILE, index=False)

def validate_password(password):
    """
    Password must contain:
    - At least 8 characters
    - One uppercase letter
    - One lowercase letter
    - One digit
    - One special character
    """
    if len(password) < 8:
        return False, "Password must be at least 8 characters long."

    if not re.search(r"[A-Z]", password):
        return False, "Password must contain at least one uppercase letter."

    if not re.search(r"[a-z]", password):
        return False, "Password must contain at least one lowercase letter."

    if not re.search(r"\d", password):
        return False, "Password must contain at least one number."

    if not re.search(r"[!@#$%^&*()_+\-=\[\]{};':\"\\|,.<>/?]", password):
        return False, "Password must contain at least one special character."

    return True, "Valid Password"

# ==============================================================================
# 2. PAGE CONFIGURATION & STATE INITIALIZATION
# ==============================================================================
st.set_page_config(page_title="Healthcare Facility Dashboard", layout="wide")

# ==============================================================================
# CUSTOM DASHBOARD UI (Animated Background + Glass Effect)
# ==============================================================================
def get_base64(file_path):
    with open(file_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

bg_img = get_base64("pic7.jpg")
sidebar_img = get_base64("pic7.jpg")   # or your image name

st.markdown(f"""
<style>

/* ===========================
Background Image
=========================== */

.stApp{{
    background:
        linear-gradient(
            rgba(0,0,0,0.20),
            rgba(0,0,0,0.20)
        ),
        url("data:image/jpeg;base64,{bg_img}");

    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;

    color:#184E3B;
}}

/* ===========================
Sidebar
=========================== */

[data-testid="stSidebar"]{{
    background: transparent !important;
}}

[data-testid="stSidebar"] > div:first-child{{
    background-image:
        linear-gradient(rgba(0,0,0,0.25), rgba(0,0,0,0.25)),
        url("data:image/jpeg;base64,{sidebar_img}") !important;

    background-size: cover !important;
    background-position: center !important;
    background-repeat: no-repeat !important;
}}

/* ===========================
Header
=========================== */

[data-testid="stHeader"]{{

background:rgba(0,0,0,0);

}}


/* ===========================
Main Block
=========================== */

.main{{

background:transparent;

}}


/* ===========================
Metric Cards
=========================== */

div[data-testid="metric-container"]{{

    background: rgba(255,255,255,0.25);

    border-radius:15px;

    padding:18px;

    border:1px solid #95D5B2;

    box-shadow:0px 4px 12px rgba(82,183,136,0.25);

    transition:0.3s;

}}

div[data-testid="metric-container"]:hover{{

    transform:translateY(-4px);

    box-shadow:0px 8px 18px rgba(82,183,136,0.35);

}}

/* ===========================
Buttons
=========================== */

.stButton>button{{

    background:#52B788;

    color:white;

    border:none;

    border-radius:10px;

    font-weight:bold;

}}

.stButton>button:hover{{

    background:#40916C;

}}

/* ===========================
Text Inputs
=========================== */

/* Text Input Boxes */
.stTextInput input {{

    background-color: white !important;

    color: black !important;

    border: 2px solid #52B788 !important;

    border-radius: 10px;

}}

/* Placeholder text */
.stTextInput input::placeholder {{

    color: #555555 !important;

}}

/* Label text */
.stTextInput label {{

    color: black !important;

    font-weight: bold;

}}


/* ===========================
Select Box
=========================== */

/* Selectbox */
.stSelectbox label,
.stRadio label,
.stMarkdown,
p,
span {{

    color: black !important;

}}

/* Selectbox input */
.stSelectbox div[data-baseweb="select"] {{

    color: black !important;

    background-color: white !important;

}}


/* ===========================
Tables
=========================== */

[data-testid="stDataFrame"]{{

background:rgba(255,255,255,.05);

border-radius:12px;

}}


/* ===========================
Titles
=========================== */

h1,h2,h3,h4{{
    color:#2D6A4F;
    font-weight:bold;
    text-shadow:none !important;
}}


/* ===========================
Floating Animated Bubbles
=========================== */

.bubble{{

position:fixed;

bottom:-120px;

width:40px;

height:40px;

border-radius:50%;

background:rgba(255,255,255,.12);

animation:rise 18s infinite;

z-index:-1;

}}

.bubble:nth-child(1){{

left:5%;

animation-duration:18s;

}}

.bubble:nth-child(2){{

left:20%;

animation-duration:22s;

width:25px;

height:25px;

}}

.bubble:nth-child(3){{

left:35%;

animation-duration:17s;

}}

.bubble:nth-child(4){{

left:55%;

animation-duration:26s;

width:18px;

height:18px;

}}

.bubble:nth-child(5){{

left:72%;

animation-duration:19s;

}}

.bubble:nth-child(6){{

left:90%;

animation-duration:24s;

width:30px;

height:30px;

}}

@keyframes rise{{

0%{{

transform:translateY(0) scale(.2);

opacity:.5;

}}

100%{{

transform:translateY(-120vh) scale(2);

opacity:0;

}}

}}

</style>

<div class="bubble"></div>
<div class="bubble"></div>
<div class="bubble"></div>
<div class="bubble"></div>
<div class="bubble"></div>
<div class="bubble"></div>

""", unsafe_allow_html=True)

# App authentication state parameters
if 'authenticated' not in st.session_state:
    st.session_state['authenticated'] = False
if 'current_user' not in st.session_state:
    st.session_state['current_user'] = None

# OTP Engine State Management Machine
if 'pwd_reset_step' not in st.session_state:
    st.session_state['pwd_reset_step'] = 1
if 'pwd_reset_user' not in st.session_state:
    st.session_state['pwd_reset_user'] = None
if 'pwd_reset_otp' not in st.session_state:
    st.session_state['pwd_reset_otp'] = None

if 'user_recover_step' not in st.session_state:
    st.session_state['user_recover_step'] = 1
if 'user_recover_phone' not in st.session_state:
    st.session_state['user_recover_phone'] = None
if 'user_recover_otp' not in st.session_state:
    st.session_state['user_recover_otp'] = None

@st.cache_data
def load_data():
    try:
        df = pd.read_csv("C:/Users/INTEL-PC/Downloads/healthcare_facility_utilization.csv")
        df['Admission_Date'] = pd.to_datetime(df['Admission_Date'])
        df['Discharge_Date'] = pd.to_datetime(df['Discharge_Date'])
        
        df['ICU_Usage'] = df['ICU_Usage'].astype(str).str.upper().str.strip().map({
            'YES': 1, 'NO': 0, '1': 1, '0': 0, '1.0': 1, '0.0': 0
        })
        df['ICU_Usage'] = df['ICU_Usage'].fillna(0).astype(int)
        df['BOR'] = (df['Occupied_Beds'] / df['Total_Beds']) * 100
        return df
    except Exception as e:
        st.error(f"Could not load data file. Error: {e}")
        return None

# ==============================================================================
# 3. AUTHENTICATION PORTAL (LOGIN / REGISTER / FORGOT)
# ==============================================================================
if not st.session_state['authenticated']:
    st.title("🏥 Healthcare Analytics Portal")
    st.markdown("Log in, register, or use secret questions verification to manage lost credentials.")
    #st.info("💡 **Standard Access Credentials** → Username: `admin` | Password: `password123` (Registered Phone: `9876543210`) ")
    
    auth_tab1, auth_tab2, auth_tab3 = st.tabs(["🔒 Secure Login", "📝 Create Account", "🔑 Forgot Credentials?"])
    
    # 3.1 Login Framework
    with auth_tab1:
        st.subheader("Login")
        login_user = st.text_input("Username", key="login_username").strip()
        login_pass = st.text_input("Password", type="password", key="login_password")
        
        if st.button("Log In", type="primary"):
            users_df = get_users_df()
            user_record = users_df[users_df['Username'] == login_user]
            
            if not user_record.empty and str(user_record.iloc[0]['Password']) == login_pass:
                # Fixed uppercase 'S' to lowercase 's'
                users_df.loc[users_df['Username'] == login_user, 'Last_Login'] = str(pd.Timestamp.now().round('s'))
                save_users_df(users_df)
                
                st.session_state['authenticated'] = True
                st.session_state['current_user'] = login_user
                st.success(f"Welcome back!")
                st.rerun()
            else:
                st.error("Invalid Username or Password.")
                
    # 3.2 Registration Framework            
    with auth_tab2:
        st.subheader("Register Profile Account")
        col_reg1, col_reg2 = st.columns(2)
        with col_reg1:
            reg_user = st.text_input("Choose Username *", key="reg_username").strip()
            reg_pass = st.text_input("Choose Password *", type="password", key="reg_password")
            reg_confirm = st.text_input("Confirm Password *", type="password", key="reg_confirm")
        with col_reg2:
            reg_first_name = st.text_input("First Name", key="reg_fname").strip()
            reg_last_name = st.text_input("Last Name", key="reg_lname").strip()
            reg_phone = st.text_input("Phone Number (digits only) *", key="reg_phone").strip()
            reg_email = st.text_input("Email ID", key="reg_email").strip()
            secret_question = st.selectbox(
                "Select Secret Question",
                [
                    "What is your mother's maiden name?",
                    "What is your favourite food?",
                    "What was your first school?",
                    "What is your pet's name?",
                    "What is your favourite colour?"
                ]
            )

            secret_answer = st.text_input("Secret Answer").strip()

        if st.button("Register"):
            users_df = get_users_df()

            email_pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'

            if not reg_user or not reg_pass or not reg_phone:
                st.warning("Username, Password, and Phone Number are required fields.")

            elif reg_user in users_df["Username"].values:
                st.error("Username already exists.")

            elif not reg_phone.isdigit() or len(reg_phone) != 10:
                st.error("Phone number must contain exactly 10 digits.")

            elif reg_phone in users_df["Phone"].astype(str).values:
                st.error("Phone number already registered.")

            elif reg_email and not re.match(email_pattern, reg_email):
                st.error("Please enter a valid email address.")

            elif reg_email and reg_email.lower() in users_df["Email"].astype(str).str.lower().values:
                st.error("Email address is already registered.")
            
            elif reg_pass != reg_confirm:
                st.error("Passwords do not match.")

            else:
                valid, message = validate_password(reg_pass)

                if not valid:
                    st.error(message)

                else:
                    new_user = pd.DataFrame([{
                        "Username": reg_user,
                        "Password": reg_pass,
                        "Secret_Question": secret_question,
                        "Secret_Answer": secret_answer.lower(),
                        "First_Name": reg_first_name if reg_first_name else "N/A",
                        "Last_Name": reg_last_name if reg_last_name else "N/A",
                        "Phone": reg_phone,
                        "Email": reg_email if reg_email else "N/A",
                        "Registration_Date": str(pd.Timestamp.now().round("s")),
                        "Last_Login": "Never",
                        "Last_Logout": "Never"
                    }])

                    users_df = pd.concat([users_df, new_user], ignore_index=True)
                    save_users_df(users_df)
                    st.success("Registration successful! Proceed to the login tab.")

    # 3.3 Dynamic OTP Verification Framework
    with auth_tab3:
        st.subheader("Account Recovery System")
        recovery_mode = st.radio("What credential do you need to recover?", ["Reset Forgotten Password", "Find Forgotten Username"])
        st.markdown("---")
        
        users_df = get_users_df()
        
        # --- FORGOT PASSWORD SUB-FLOW ---
        if recovery_mode == "Reset Forgotten Password":
            username = st.text_input("Enter Username")
            if username:
                user = users_df[users_df["Username"] == username]
                if user.empty:
                    st.error("Username not found.")
                else:
                    st.info(user.iloc[0]["Secret_Question"])
                    answer = st.text_input("Secret Answer", key="forgot_password_answer")
                    new_password = st.text_input(
                        "New Password",
                        type="password",
                        key="reset_new_password"
                    )
                    confirm_password = st.text_input(
                        "Confirm Password",
                        type="password",
                        key="reset_confirm_password"
                    )
                    if st.button("Reset Password"):
                        if answer.lower() != str(user.iloc[0]["Secret_Answer"]).lower():
                            st.error("Incorrect Secret Answer.")
                        elif new_password != confirm_password:
                            st.error("Passwords do not match.")
                        else:
                            valid, message = validate_password(new_password)
                            if not valid:
                                st.error(message)
                            else:
                                users_df.loc[
                                    users_df["Username"] == username,
                                    "Password"
                                ] = new_password
                                save_users_df(users_df)
                                st.success("Password reset successfully.")
                        
        # --- FORGOT USERNAME SUB-FLOW ---
        else:
            st.subheader("Recover Username")
            first_name = st.text_input("Enter First Name")
            last_name = st.text_input("Enter Last Name")
            if first_name and last_name:
                user = users_df[
                (users_df["First_Name"].str.lower() == first_name.lower()) &
                (users_df["Last_Name"].str.lower() == last_name.lower())
                ]
                if user.empty:
                    st.error("No matching user found.")
                else:
                    st.info(user.iloc[0]["Secret_Question"])
                    answer = st.text_input("Secret Answer", key="forgot_username_answer")
                if st.button("Recover Username"):
                    if answer.lower() == str(user.iloc[0]["Secret_Answer"]).lower():
                        st.success("Identity Verified!")
                        st.info(f"Your Username is: **{user.iloc[0]['Username']}**")
                    else:
                        st.error("Incorrect Secret Answer.")

# ==============================================================================
# 4. MAIN DASHBOARD PIPELINE (PROTECTED ROUTE)
# ==============================================================================
else:
    df = load_data()

    filtered_df = df.copy()

    department = sorted(df["Department"].unique())
    bed_type = sorted(df["Bed_Type"].unique())
    doctor = sorted(df["Doctor_Assigned"].unique())

    if df is not None:
        users_df = get_users_df()
        current_profile = users_df[users_df['Username'] == st.session_state['current_user']].iloc[0]
        
        st.sidebar.markdown("""
        <div style="
        background:white;
        padding:20px;
        border-radius:20px;
        text-align:center;
        box-shadow:0 5px 15px rgba(0,0,0,.08);
        margin-bottom:20px;
        ">

        <h1 style="font-size:60px;margin:0;">🏥</h1>

        <h3 style="color:#2D6A4F;margin-bottom:0;">
        Healthcare Portal
        </h3>

        <hr>

        <h4 style="margin-bottom:0;">
        Welcome
        </h4>

        <p style="font-size:22px;color:#40916C;font-weight:bold;">
        {}
        </p>

        <p style="color:gray;">
        {}
        </p>

        </div>
        """.format(
        current_profile["First_Name"],
        current_profile["Username"]
        ), unsafe_allow_html=True)
        
        logout = st.sidebar.button(
            "🚪 Logout",
            use_container_width=True
        )

        if logout:

            users_df = get_users_df()

            users_df.loc[
                users_df["Username"] ==
                st.session_state["current_user"],
                "Last_Logout"
            ] = str(pd.Timestamp.now().round("s"))

            save_users_df(users_df)

            st.session_state["authenticated"] = False
            st.session_state["current_user"] = None

            st.rerun()
            
        st.sidebar.markdown("---")

        st.sidebar.markdown("---")

# =====================================
# DASHBOARD FILTERS (SLICERS)
# =====================================
        

# =====================================

        nav_options = [
            "Data Overview",
            "Exploratory Data Analysis",
            "Model Training & Evaluation",
            "Facility Clustering",
            "⚙️ Edit Profile"
        ]
        
        if st.session_state['current_user'] == 'admin':
            nav_options.append("🔐 System Audit Logs")
            
            
        page = st.sidebar.radio("Navigate Dashboard", nav_options)
        st.markdown("""
        <h1 style="
        font-size:42px;
        color:#2D6A4F;
        margin-bottom:0px;
        ">
        🏥 Healthcare Facility Dashboard
        </h1>

        <p style="
        font-size:18px;
        color:#777;
        margin-top:0px;
        ">
        Smart Healthcare Analytics & Resource Management System
        </p>
        """, unsafe_allow_html=True)

        # ==============================================================================
        # 4.1 DATA OVERVIEW
        # ==============================================================================
        if page == "Data Overview":

            from datetime import datetime

    # ---------------- HEADER ---------------- #

            col1, col2 = st.columns([5,1])

            with col1:
                st.markdown("""
                <h1 style='color:#2D6A4F;margin-bottom:0px;'>
                🏥 Healthcare Dashboard
                </h1>

                <p style='font-size:18px;color:#6c757d;margin-top:0px;'>
                Healthcare Facility Utilization Analytics
                </p>
                """, unsafe_allow_html=True)

            with col2:
                st.markdown(f"""
                <div style="
                    background:white;
                    padding:15px;
                    border-radius:15px;
                    text-align:center;
                    box-shadow:0 3px 10px rgba(0,0,0,.12);
                ">
                <h4 style="margin:0;color:#2D6A4F;">
                {datetime.now().strftime("%d %b %Y")}
                </h4>

                <small>{datetime.now().strftime("%I:%M %p")}</small>

                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

    # ---------------- KPI CARDS ---------------- #

            c1,c2,c3,c4,c5,c6 = st.columns(6)

            total_beds = int(filtered_df["Total_Beds"].sum())
            occupied = int(filtered_df["Occupied_Beds"].sum())
            available = total_beds-occupied

            avg_resource = filtered_df["Resource_Usage_%"].mean()

            total_doctors = filtered_df["Doctor_Assigned"].nunique()

            def card(title,value,color,icon):

                st.markdown(f"""
                <div style="
                background:white;
                border-left:8px solid {color};
                border-radius:18px;
                padding:20px;
                box-shadow:0 5px 15px rgba(0,0,0,.08);
                ">
                <h1 style="margin:0">{icon}</h1>

                <h2 style="color:{color};margin-bottom:5px;">
                {value}
                </h2>

                <p style="font-size:18px;color:#555;">
                {title}
                </p>

                </div>
                """,unsafe_allow_html=True)

            with c1:
                card("Records",filtered_df.shape[0],"#40916C","📄")

            with c2:
                card("Total Beds",total_beds,"#2D6A4F","🛏️")

            with c3:
                card("Occupied",occupied,"#E76F51","🏥")

            with c4:
                card("Available",available,"#34EE43","👨‍⚕️")

            with c5:
                card("Doctors",total_doctors,"#4361EE","👨‍⚕️")

            with c6:
                card("Resource Usage",f"{avg_resource:.1f}%","#F4A261","📈")

            st.markdown("<br>",unsafe_allow_html=True)

            c1, c2 = st.columns(2)

            with c1:

                occupancy = (
                occupied / total_beds
                ) * 100

                st.markdown("### 🛏️ Bed Occupancy")

                st.progress(int(occupancy))

                st.write(f"{occupancy:.1f}%")

            with c2:

                st.markdown("### ⚙️ Resource Utilization")

                st.progress(int(avg_resource))

                st.write(f"{avg_resource:.1f}%")

            st.markdown("""
            <div style="
            background:white;
            padding:20px;
            border-radius:18px;
            box-shadow:0 5px 15px rgba(0,0,0,.08);
            margin-bottom:20px;
            ">
            <h3 style="color:#2D6A4F;">🔎 Dashboard Filters</h3>
            </div>
            """, unsafe_allow_html=True)

            f1, f2, f3, f4, f5 = st.columns(5)

            with f1:
                department = st.multiselect(
                    "Department",
                    sorted(df["Department"].unique()),
                    default=department,
                    key="dash_department"
                )

            with f2:
                bed_type = st.multiselect(
                    "Bed Type",
                    sorted(df["Bed_Type"].unique()),
                    default=bed_type,
                    key="dash_bed"
                )

            with f3:
                icu = st.selectbox(
                    "ICU Usage",
                    ["All", "Yes", "No"],
                    key="dash_icu"
                )

            with f4:
                doctor = st.multiselect(
                    "Doctor",
                    sorted(df["Doctor_Assigned"].unique()),
                    default=doctor,
                    key="dash_doctor"
                )

            with f5:
                status = st.selectbox(
                    "Sort Records",
                    ["Newest", "Highest Resource", "Lowest Resource"],
                    key="sort_option"
                )

            filtered_df = df.copy()

            filtered_df = filtered_df[
                filtered_df["Department"].isin(department)
            ]

            filtered_df = filtered_df[
                filtered_df["Bed_Type"].isin(bed_type)
            ]

            filtered_df = filtered_df[
                filtered_df["Doctor_Assigned"].isin(doctor)
            ]

            if icu == "Yes":
                filtered_df = filtered_df[
                filtered_df["ICU_Usage"] == 1
            ]
            elif icu == "No":
                filtered_df = filtered_df[
                filtered_df["ICU_Usage"] == 0
            ]

            b1, b2 = st.columns([1,1])

            with b1:
                apply = st.button("✅ Apply Filters", use_container_width=True)

            with b2:
                reset = st.button("🔄 Reset Filters", use_container_width=True)
            
            if reset:
                for key in [
                    "dash_department",
                    "dash_bed",
                    "dash_doctor",
                    "dash_icu",
                    "sort_option",
                ]:
                    if key in st.session_state:
                        del st.session_state[key]

                st.rerun()

            if status == "Highest Resource":
                filtered_df = filtered_df.sort_values(
                    "Resource_Usage_%",
                    ascending=False
                )

            elif status == "Lowest Resource":
                filtered_df = filtered_df.sort_values(
                    "Resource_Usage_%",
                    ascending=True
                )

            else:
                filtered_df = filtered_df.sort_values(
                    "Admission_Date",
                    ascending=False
                )
            
            left,right = st.columns([1.6,1])

            with left:

                st.markdown("""
                <div style="
                background:white;
                padding:20px;
                border-radius:18px;
                box-shadow:0 5px 15px rgba(0,0,0,.08);
                ">
                <h3 style="color:#2D6A4F;">
                📋 Data Preview
                </h3>
                </div>
                """,unsafe_allow_html=True)

                st.dataframe(
                    filtered_df[
                        [
                            "Doctor_Assigned",
                            "Department",
                            "Bed_Type",
                            "Occupied_Beds",
                            "Total_Beds",
                            "Resource_Usage_%"
                        ]
                    ].head(12),
                    use_container_width=True,
                    height=420
                )

            with right:

                dept = filtered_df.groupby("Department")["Occupied_Beds"].sum().reset_index()

                fig = px.bar(
                    dept,
                    x="Department",
                    y="Occupied_Beds",
                    color="Department",
                    title="Department Occupancy"
                )

                fig.update_layout(
                    plot_bgcolor="white",
                    paper_bgcolor="white",
                    height=450,
                    title_x=.25
                )

                st.plotly_chart(fig,use_container_width=True)

            st.markdown("---")
            st.subheader("🏥 Most Used Medical Equipment")

            equipment = (
                filtered_df["Equipment_Used"]
                .fillna("Unknown")
                .value_counts()
                .head(10)
                .reset_index()
            )
            
            equipment.columns = ["Equipment", "Usage Count"]

            fig_equipment = px.bar(
                equipment,
                x="Equipment",
                y="Usage Count",
                color="Equipment",
                title="Most Used Medical Equipment"
            )

            fig_equipment.update_layout(
                xaxis_title="Equipment",
                yaxis_title="Number of Uses",
                plot_bgcolor="white",
                paper_bgcolor="white"
            )

            st.plotly_chart(fig_equipment, use_container_width=True)

            st.markdown("""
            <h3 style="color:#2D6A4F;">
            📢 Recent Activity
            </h3>
            """, unsafe_allow_html=True)

            recent = filtered_df.sort_values(
                "Admission_Date",
                ascending=False
            ).head(5)

            for _, row in recent.iterrows():

                st.markdown(f"""
                <div style="
                background:white;
                padding:15px;
                border-radius:15px;
                margin-bottom:10px;
                box-shadow:0 4px 12px rgba(0,0,0,.08);
                border-left:6px solid #52B788;
                ">

                <h4 style="margin:0;color:#2D6A4F;">
                👨‍⚕️ {row['Doctor_Assigned']}
                </h4>

                <p style="margin:4px 0;">
                <b>Department:</b> {row['Department']}
                </p>

                <p style="margin:4px 0;">
                <b>Occupied Beds:</b> {row['Occupied_Beds']}
                </p>
                
                <small style="color:gray;">
                📅 {row['Admission_Date']}
                </small>

                </div>
                """, unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("""
            <center>
            Healthcare Analytics Dashboard<br>
            Developed using Streamlit | Machine Learning | Plotly
            </center>
            """, unsafe_allow_html=True)

        # ==============================================================================
        # 4.2 EXPLORATORY DATA ANALYSIS (EDA)
        # ==============================================================================
        elif page == "Exploratory Data Analysis":
            st.header("📊 Interactive Visualizations")
            col1, col2 = st.columns(2)
            
            with col1:
                bor_per_dept = filtered_df.groupby('Department')['BOR'].mean().reset_index()
                fig1 = px.bar(bor_per_dept, x='Department', y='BOR', title="Average Bed Occupancy Rate (BOR) by Department", color='Department')
                st.plotly_chart(fig1, use_container_width=True)
                
                icu_by_dept = filtered_df.groupby('Department')['ICU_Usage'].mean().reset_index()
                fig2 = px.bar(icu_by_dept, x='Department', y='ICU_Usage', title="Average ICU Usage Rate by Department", color='Department')
                st.plotly_chart(fig2, use_container_width=True)

            with col2:
                avg_len_durn = filtered_df.groupby('Department')['Treatment_Duration_Days'].mean().reset_index()
                fig3 = px.bar(avg_len_durn, x='Department', y='Treatment_Duration_Days', title="Avg Treatment Duration (Days) by Department", color='Department')
                st.plotly_chart(fig3, use_container_width=True)
                
                avg_reso_dept = filtered_df.groupby('Department')['Resource_Usage_%'].mean().reset_index()
                fig4 = px.bar(avg_reso_dept, x='Department', y='Resource_Usage_%', title="Avg Resource Usage (%) by Department", color='Department')
                st.plotly_chart(fig4, use_container_width=True)

            st.markdown("---")
            st.subheader("Resource Usage Analysis")
            resource_pivot = filtered_df.pivot_table(index='Department', columns='Bed_Type', values='Resource_Usage_%', aggfunc='mean')
            fig_heat = px.imshow(resource_pivot, text_auto=".1f", aspect="auto", title="Average Resource Usage (%) by Department and Bed Type", color_continuous_scale='Viridis')
            st.plotly_chart(fig_heat, use_container_width=True)

        # ==============================================================================
        # 4.3 MODEL TRAINING & EVALUATION
        # ==============================================================================
        elif page == "Model Training & Evaluation":
            st.header("🤖 Predictive Modeling")
            model_choice = st.selectbox("Select a Model to Evaluate", [
                "Linear Regression (Predict Duration)", "Logistic Regression (Predict ICU Usage)", 
                "Decision Tree Regressor (Predict Resource Usage)", "Decision Tree Classifier (Predict Resource Class)",
                "Random Forest Regressor (Predict Resource Usage)"
            ])
            
            st.markdown("---")
            
            if model_choice == "Linear Regression (Predict Duration)":
                X = filtered_df[['Occupied_Beds']]
                y = filtered_df['Treatment_Duration_Days']
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                model = LinearRegression().fit(X_train, y_train)
                preds = model.predict(X_test)
                st.metric("R² Score", f"{r2_score(y_test, preds):.4f}")
                fig = px.scatter(x=X_test['Occupied_Beds'], y=y_test, labels={'x': 'Occupied Beds', 'y': 'Duration Days'})
                fig.add_scatter(x=X_test['Occupied_Beds'], y=preds, mode='lines', name='Regression Line', line=dict(color='red'))
                st.plotly_chart(fig, use_container_width=True)

            elif model_choice == "Logistic Regression (Predict ICU Usage)":
                X = filtered_df[['Resource_Usage_%']]
                y = filtered_df['ICU_Usage']
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                model = LogisticRegression().fit(X_train, y_train)
                probs = model.predict_proba(X_test)[:, 1]
                st.metric("Accuracy Score", f"{accuracy_score(y_test, model.predict(X_test)):.4f}")
                fig = px.scatter(x=X_test['Resource_Usage_%'], y=y_test, labels={'x': 'Resource Usage %', 'y': 'ICU Usage Probability'})
                fig.add_scatter(x=X_test['Resource_Usage_%'], y=probs, mode='markers', name='Probability', marker=dict(color='red', size=5))
                st.plotly_chart(fig, use_container_width=True)

            elif model_choice == "Decision Tree Regressor (Predict Resource Usage)":
                X = filtered_df[['Total_Beds','Occupied_Beds','Staff_On_Duty','Treatment_Duration_Days']]
                y = filtered_df['Resource_Usage_%']
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                tree_model = DecisionTreeRegressor(random_state=42, max_depth=3).fit(X_train, y_train)
                st.metric("R² Score", f"{r2_score(y_test, tree_model.predict(X_test)):.4f}")

            elif model_choice == "Decision Tree Classifier (Predict Resource Class)":
                X = filtered_df[['Total_Beds','Occupied_Beds','Staff_On_Duty','Treatment_Duration_Days']]
                y = pd.cut(filtered_df['Resource_Usage_%'], bins=[-1, 50, 80, 100], labels=['Low','Medium','High'])
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                tree_model = DecisionTreeClassifier(random_state=42, max_depth=3).fit(X_train, y_train)
                st.metric("Accuracy Score", f"{accuracy_score(y_test, tree_model.predict(X_test)):.4f}")
                st.code(classification_report(y_test, tree_model.predict(X_test)))

            elif model_choice == "Random Forest Regressor (Predict Resource Usage)":
                X = filtered_df[['BOR']]
                y = filtered_df['Resource_Usage_%']
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                model = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train, y_train)
                st.metric("R² Score", f"{r2_score(y_test, model.predict(X_test)):.4f}")

        # ==============================================================================
        # 4.4 FACILITY CLUSTERING
        # ==============================================================================
        elif page == "Facility Clustering":
            st.header("🧩 Facility Segmentation via K-Means Clustering")
            num_clusters = st.slider("Select Number of Clusters (k)", min_value=2, max_value=6, value=3)
            
            cluster_features = ['BOR', 'Resource_Usage_%']
            cluster_data = filtered_df[['Department', 'Bed_Type'] + cluster_features].dropna()
            
            scaled_features = StandardScaler().fit_transform(cluster_data[cluster_features])
            cluster_data['Cluster'] = KMeans(n_clusters=num_clusters, random_state=42, n_init=10).fit_predict(scaled_features)
            cluster_data['Cluster'] = cluster_data['Cluster'].astype(str)
            
            fig_cluster = px.scatter(cluster_data, x='BOR', y='Resource_Usage_%', color='Cluster', hover_data=['Department', 'Bed_Type'])
            st.plotly_chart(fig_cluster, use_container_width=True)
            st.dataframe(cluster_data.groupby('Cluster')[cluster_features].mean().reset_index(), use_container_width=True)

        # ==============================================================================
        # 4.5 USER PROFILE MANAGEMENT (EDIT INTERFACE)
        # ==============================================================================
        elif page == "⚙️ Edit Profile":
            st.header("👤 Manage Profile Parameters")
            current_username = st.session_state['current_user']
            users_df = get_users_df()
            user_idx = users_df[users_df['Username'] == current_username].index[0]
            
            with st.form("profile_edit_form"):
                st.subheader("Edit Demographics")
                col_edit1, col_edit2 = st.columns(2)
                
                with col_edit1:
                    new_fname = st.text_input("First Name", value=str(users_df.at[user_idx, 'First_Name']))
                    new_lname = st.text_input("Last Name", value=str(users_df.at[user_idx, 'Last_Name']))
                with col_edit2:
                    new_phone = st.text_input("Phone Number", value=str(users_df.at[user_idx, 'Phone']))
                    new_email = st.text_input("Email ID", value=str(users_df.at[user_idx, 'Email']))
                    
                st.markdown("---")
                st.subheader("Change System Password")
                new_pwd = st.text_input("New Password (Leave blank to keep current)", type="password")
                confirm_pwd = st.text_input("Confirm New Password", type="password")
                
                submit_change = st.form_submit_button("Save Modification Profile Changes")
                
                if submit_change:
                    users_df.at[user_idx, 'First_Name'] = new_fname.strip() if new_fname else "N/A"
                    users_df.at[user_idx, 'Last_Name'] = new_lname.strip() if new_lname else "N/A"
                    users_df["Phone"] = users_df["Phone"].astype(str)
                    users_df.at[user_idx, "Phone"] = new_phone.strip() if new_phone else ""
                    users_df.at[user_idx, 'Email'] = new_email.strip() if new_email else "N/A"

                    if new_pwd:
                        valid, message = validate_password(new_pwd)

                        if not valid:
                            st.error(message)

                        elif new_pwd != confirm_pwd:
                            st.error("Passwords do not match.")

                        else:
                            users_df.at[user_idx, 'Password'] = new_pwd

                    save_users_df(users_df)
                    st.success("Profile updated successfully!")

        # ==============================================================================
        # 4.6 SYSTEM AUDIT LOGS
        # ==============================================================================
        elif page == "🔐 System Audit Logs":
            st.header("🔒 System User Audit Registry")
            st.markdown("Restricted view revealing demographics and login metadata inside `users_db.csv`.")
            users_df = get_users_df()
            st.dataframe(users_df, use_container_width=True)