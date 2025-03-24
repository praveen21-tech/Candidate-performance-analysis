
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import os
import json
import requests
from sklearn.cluster import KMeans
from fpdf import FPDF

# Load API Key from Environment Variables
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
HEADERS = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"} if GROQ_API_KEY else None

# Admin Credentials (Consider replacing with a database in production)
admin_users = {"admin": "password123"}

# Session State for Authentication & Navigation
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "current_task" not in st.session_state:
    st.session_state.current_task = None

# Function: Admin Login
def login():
    st.sidebar.title("🔐 Admin Login")
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")
    if st.sidebar.button("Login"):
        if username in admin_users and admin_users[username] == password:
            st.session_state.authenticated = True
            st.sidebar.success("✅ Login Successful!")
            st.rerun()
        else:
            st.sidebar.error("❌ Invalid Credentials")

# AI Recommendation Function using GROQ API (LLaMA 3.3 70B Model)
def get_ai_recommendations(candidate_data):
    prompt = f"Given the following student performance data: {candidate_data}, recommend appropriate courses based on their performance level."

    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }

    response = requests.post(GROQ_API_URL, headers=HEADERS, data=json.dumps(payload))

    if response.status_code == 200:
        ai_response = response.json()
        recommendations = ai_response.get("choices", [])[0].get("message", {}).get("content", "")
        return recommendations
    else:
        st.error(f"Error: {response.status_code} - {response.text}")
        return None

# Function to display the Student Performance Dashboard with Key Insights
def student_performance_dashboard(df):
    st.write("## 📊 Student Performance Dashboard")
    
    # Key Insights
    total_students = df["Candidate Name"].nunique()
    strong_category_count = df[df["Performance Category"] == "Strong"].shape[0]
    average_category_count = df[df["Performance Category"] == "Average"].shape[0]
    weak_category_count = df[df["Performance Category"] == "Weak"].shape[0]
    
    highest_marks = df["Mark"].max()
    lowest_marks = df["Mark"].min()
    average_marks = df["Mark"].mean()

    # Display Insights
    st.write(f"### Key Insights")
    st.write(f"- Total Students: {total_students}")
    st.write(f"- Strong Category Students: {strong_category_count}")
    st.write(f"- Average Category Students: {average_category_count}")
    st.write(f"- Weak Category Students: {weak_category_count}")
    st.write(f"- Highest Marks: {highest_marks}")
    st.write(f"- Lowest Marks: {lowest_marks}")
    st.write(f"- Average Marks: {average_marks:.2f}")

    # Performance Category Breakdown
    st.write("### Performance Category Breakdown")
    performance_category = df["Performance Category"].value_counts()
    st.bar_chart(performance_category, use_container_width=True)

    # Performance Trends by Attempts
    st.write("### Performance Trends by Attempts")
    attempts_performance = df.groupby("Attempts")["Mark"].mean()
    st.line_chart(attempts_performance, use_container_width=True)

# Title
st.title("📊 AI-Powered Student Performance Analysis & Recommendation Tool")

# Authentication Check
if not st.session_state.authenticated:
    login()
else:
    # Centralized file uploader (drag-and-drop feature removed)
    uploaded_file = st.file_uploader("Upload Student Data (CSV/XLSX)", type=["csv", "xlsx"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
        st.success("✅ File Uploaded Successfully!")
        st.write("### 🔍 Data Preview")
        st.dataframe(df.head())

        # Required Columns Validation
        required_columns = ["Candidate Name", "Candidate Email", "Course Name", "Mark"]
        if not all(col in df.columns for col in required_columns):
            st.error("❌ Missing required columns! Please check your file.")
        else:
            if "Attempts" not in df.columns:
                df["Attempts"] = np.random.randint(1, 4, size=len(df))

            # Add Performance Category
            df["Performance Category"] = pd.qcut(df["Mark"], q=3, labels=["Weak", "Average", "Strong"])

            # Show Student Performance Dashboard with Key Insights
            student_performance_dashboard(df)

            # Dashboard Buttons
            if st.sidebar.button("📊 Analyze Performance"):
                st.session_state.current_task = "Analyze Performance"
            if st.sidebar.button("✅ Categorize Students"):
                st.session_state.current_task = "Categorize Students"
            if st.sidebar.button("🎯 Recommend Courses"):
                st.session_state.current_task = "Recommend Courses"
            if st.sidebar.button("🤝 Mentor-Mentee Pairs"):
                st.session_state.current_task = "Mentor-Mentee Pairs"
            if st.sidebar.button("📥 Download Reports"):
                st.session_state.current_task = "Download Reports"

            # Display Task Based on Selection
            if st.session_state.current_task == "Analyze Performance":
                st.write("## 📊 Performance Analysis")
                st.write("### 📈 Student Performance Trends")

                # Create a bar chart for distribution of marks
                marks_distribution = df["Mark"].value_counts().sort_index()
                st.bar_chart(marks_distribution, use_container_width=True)

                # Performance trends over attempts
                attempts_performance = df.groupby("Attempts")["Mark"].mean()
                st.line_chart(attempts_performance, use_container_width=True)

                # Performance by category
                performance_category = df["Performance Category"].value_counts()
                st.bar_chart(performance_category, use_container_width=True)

                st.write("### 📉 Performance by Category")
                st.area_chart(performance_category, use_container_width=True)

                st.write("### 📈 Marks by Attempts")
                marks_by_attempt = df.groupby("Attempts")["Mark"].mean()
                st.line_chart(marks_by_attempt, use_container_width=True)

            elif st.session_state.current_task == "Categorize Students":
                st.write("## ✅ Student Categories")
                category_filter = st.multiselect("Select Performance Categories", ["Weak", "Average", "Strong"], default=["Weak", "Average", "Strong"])
                course_filter = st.multiselect("Select Courses", ["All Courses"] + df["Course Name"].unique().tolist(), default=["All Courses"])

                if "All Courses" in course_filter:
                    filtered_df = df[df["Performance Category"].isin(category_filter)]
                else:
                    filtered_df = df[df["Performance Category"].isin(category_filter) & df["Course Name"].isin(course_filter)]

                st.dataframe(filtered_df)

            elif st.session_state.current_task == "Recommend Courses":
                st.write("## 🎯 AI-Based Course Recommendations")

                selected_subject = st.selectbox("Select Course", ["All Courses"] + df["Course Name"].unique().tolist())

                if selected_subject != "All Courses":
                    df_filtered = df[df["Course Name"] == selected_subject]
                else:
                    df_filtered = df

                selected_candidate = st.selectbox("Select Candidate", df_filtered["Candidate Name"].unique())

                candidate_data = df_filtered[df_filtered["Candidate Name"] == selected_candidate].iloc[0]
                ai_recommendations = get_ai_recommendations(candidate_data)

                if ai_recommendations:
                    st.write("### 📋 Candidate AI-Based Course Recommendations")
                    st.markdown(ai_recommendations)
                else:
                    st.warning("No AI-based recommendations available at the moment.")

            elif st.session_state.current_task == "Mentor-Mentee Pairs":
                st.write("## 🤝 Mentor-Mentee Matching")

                selected_subject = st.selectbox("Select Course", df["Course Name"].unique())

                # Select mentors (Strong performers) and mentees (Weak performers) for the selected course
                mentors = df[(df["Course Name"] == selected_subject) & (df["Performance Category"] == "Strong")]
                mentees = df[(df["Course Name"] == selected_subject) & (df["Performance Category"] == "Weak")]

                # Ensure we don't pair the same person as both mentor and mentee
                mentors = mentors[~mentors["Candidate Name"].isin(mentees["Candidate Name"])]

                # Separate by attempt (group by 'Attempts')
                attempts = sorted(df["Attempts"].unique())

                all_pairs = []

                for attempt in attempts:
                    st.write(f"### Attempt {attempt}")
                    
                    mentors_attempt = mentors[mentors["Attempts"] == attempt]
                    mentees_attempt = mentees[mentees["Attempts"] == attempt]

                    if not mentors_attempt.empty and not mentees_attempt.empty:
                        mentor_mentee_pairs = []

                        # Ensure that each mentor is matched to only one mentee in this attempt
                        for i, mentor in mentors_attempt.iterrows():
                            if not mentees_attempt.empty:
                                mentee = mentees_attempt.iloc[0]
                                mentor_mentee_pairs.append((mentor["Candidate Name"], mentee["Candidate Name"]))
                                mentees_attempt = mentees_attempt.drop(mentees_attempt.index[0])  # Remove the mentee from the list after pairing

                        # Add pairs for this attempt to the overall list
                        all_pairs.extend(mentor_mentee_pairs)

                        # Display the mentor-mentee pairs for this attempt
                        if mentor_mentee_pairs:
                            pairs_df = pd.DataFrame(mentor_mentee_pairs, columns=["Mentor", "Mentee"])
                            st.write("#### Mentor-Mentee Pairs")
                            st.dataframe(pairs_df)
                        else:
                            st.warning(f"No mentor-mentee pairs created for Attempt {attempt}.")
                    else:
                        st.warning(f"Not enough mentors or mentees available for Attempt {attempt}.")

                # If no mentor-mentee pairs were created across attempts
                if not all_pairs:
                    st.warning("No mentor-mentee pairs were created.")

            elif st.session_state.current_task == "Download Reports":
                st.write("### 📥 Generate & Download Student Report")
                selected_student = st.selectbox("Select a Student", df["Candidate Name"].unique())
                course = st.selectbox("Select Course", ["All Courses"] + df["Course Name"].unique().tolist())

                student_data = df[df["Candidate Name"] == selected_student]

                if st.button("Generate Report"):
                    pdf = FPDF()
                    pdf.add_page()
                    pdf.set_font("Arial", style='B', size=16)
                
                    # Title and Header for the Report
                    pdf.cell(200, 10, "Sri Manakula Vinayagar Engineering College, Puducherry", ln=True, align='C')
                    pdf.ln(10)
                    pdf.set_font("Arial", size=12)
                    pdf.cell(200, 10, f"Student Report Card", ln=True, align='C')
                    pdf.ln(10)
                    pdf.cell(200, 10, f"Name: {selected_student}", ln=True)
                    pdf.cell(200, 10, f"Course: {course}", ln=True)
                    pdf.ln(5)

                    # Table with Columns: Subject, Attempts, Marks, Grade
                    pdf.set_font("Arial", size=12)
                    pdf.cell(65, 10, "Subject", border=1, align='C')
                    pdf.cell(40, 10, "Attempts", border=1, align='C')
                    pdf.cell(40, 10, "Marks", border=1, align='C')
                    pdf.cell(40, 10, "Grade", border=1, align='C')
                    pdf.ln()

                    # Add student data to the table
                    for _, row in student_data.iterrows():
                        pdf.cell(65, 10, row["Course Name"], border=1, align='C')
                        pdf.cell(40, 10, str(row["Attempts"]), border=1, align='C')
                        pdf.cell(40, 10, str(row["Mark"]), border=1, align='C')
                        pdf.cell(40, 10, row["Performance Category"], border=1, align='C')
                        pdf.ln()

                    # Output the PDF
                    pdf.output("student_report.pdf")
                    with open("student_report.pdf", "rb") as f:
                        st.download_button(
                            label="Download Report",
                            data=f,
                            file_name="student_report.pdf",
                            mime="application/pdf"
                        )