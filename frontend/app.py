"""Streamlit frontend for WorkforceIQ."""
import requests
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="WorkforceIQ",
    page_icon="📊",
    layout="wide",
)

# Dashboard header
st.title("WorkforceIQ")
st.subheader("Employee Attrition Prediction & Analytics")

st.markdown(
    """
    Predict employee attrition using a Machine Learning model.
    Enter employee information below to generate an attrition prediction.
    """
)

st.divider()

# Employee input form
st.header("Employee Information")

with st.form("employee_prediction_form"):

    st.subheader("Personal Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input(
            "Age",
            min_value=18,
            max_value=60,
            value=35,
        )

    with col2:
        gender = st.selectbox(
            "Gender",
            ["Male", "Female"],
        )

    with col3:
        marital_status = st.selectbox(
            "Marital Status",
            ["Single", "Married", "Divorced"],
            index=1,
        )

    st.subheader("Job Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        department = st.selectbox(
            "Department",
            [
                "Research & Development",
                "Sales",
                "Human Resources",
            ],
        )

    with col2:
        job_role = st.selectbox(
            "Job Role",
            [
                "Research Scientist",
                "Laboratory Technician",
                "Manufacturing Director",
                "Healthcare Representative",
                "Manager",
                "Sales Executive",
                "Sales Representative",
                "Research Director",
                "Human Resources",
            ],
        )

    with col3:
        business_travel = st.selectbox(
            "Business Travel",
            [
                "Travel_Rarely",
                "Travel_Frequently",
                "Non-Travel",
            ],
        )

    col1, col2, col3 = st.columns(3)

    with col1:
        job_level = st.selectbox(
            "Job Level",
            [1, 2, 3, 4, 5],
            index=1,
        )

    with col2:
        monthly_income = st.number_input(
            "Monthly Income",
            min_value=1,
            value=5000,
        )

    with col3:
        overtime = st.selectbox(
            "OverTime",
            ["No", "Yes"],
        )

    st.subheader("Work Environment")

    col1, col2, col3 = st.columns(3)

    with col1:
        environment_satisfaction = st.selectbox(
            "Environment Satisfaction",
            [1, 2, 3, 4],
            index=2,
        )

    with col2:
        job_satisfaction = st.selectbox(
            "Job Satisfaction",
            [1, 2, 3, 4],
            index=2,
        )

    with col3:
        work_life_balance = st.selectbox(
            "Work Life Balance",
            [1, 2, 3, 4],
            index=2,
        )
        
    
    st.subheader("Education & Professional Details")

    col1, col2, col3 = st.columns(3)

    with col1:
        education = st.selectbox("Education", [1, 2, 3, 4, 5])

    with col2:
        education_field = st.selectbox(
            "Education Field",
            [
                "Life Sciences",
                "Medical",
                "Marketing",
                "Technical Degree",
                "Other",
                "Human Resources",
            ],
        )

    with col3:
        job_involvement = st.selectbox(
            "Job Involvement", [1, 2, 3, 4]
        )

    col1, col2, col3 = st.columns(3)

    with col1:
        performance_rating = st.selectbox(
            "Performance Rating", [3, 4]
        )

    with col2:
        relationship_satisfaction = st.selectbox(
            "Relationship Satisfaction", [1, 2, 3, 4]
        )

    with col3:
        stock_option_level = st.selectbox(
            "Stock Option Level", [0, 1, 2, 3]
        )

    st.subheader("Compensation & Work Details")

    col1, col2, col3 = st.columns(3)

    with col1:
        daily_rate = st.number_input("Daily Rate", min_value=1, value=1100)

    with col2:
        hourly_rate = st.number_input("Hourly Rate", min_value=1, value=65)

    with col3:
        monthly_rate = st.number_input("Monthly Rate", min_value=1, value=15000)

    col1, col2, col3 = st.columns(3)

    with col1:
        distance_from_home = st.number_input(
            "Distance From Home", min_value=1, value=10
        )

    with col2:
        total_working_years = st.number_input(
            "Total Working Years", min_value=0, value=10
        )

    with col3:
        num_companies_worked = st.number_input(
            "Number of Companies Worked", min_value=0, value=2
        )

    col1, col2, col3 = st.columns(3)

    with col1:
        percent_salary_hike = st.number_input(
            "Percent Salary Hike", min_value=0, max_value=100, value=15
        )

    with col2:
        training_times_last_year = st.number_input(
            "Training Times Last Year", min_value=0, value=3
        )

    with col3:
        years_at_company = st.number_input(
            "Years at Company", min_value=0, value=5
        )

    st.subheader("Career Progression")

    col1, col2, col3 = st.columns(3)

    with col1:
        years_in_current_role = st.number_input(
            "Years in Current Role", min_value=0, value=3
        )

    with col2:
        years_since_last_promotion = st.number_input(
            "Years Since Last Promotion", min_value=0, value=1
        )

    with col3:
        years_with_curr_manager = st.number_input(
            "Years With Current Manager", min_value=0, value=3
        )


    predict_button = st.form_submit_button(
        "Predict Attrition",
        use_container_width=True,
    )

    
if predict_button:

    # Prepare employee input payload
    employee_data = {
        "Age": age,
        "BusinessTravel": business_travel,
        "DailyRate": daily_rate,
        "Department": department,
        "DistanceFromHome": distance_from_home,
        "Education": education,
        "EducationField": education_field,
        "EnvironmentSatisfaction": environment_satisfaction,
        "Gender": gender,
        "HourlyRate": hourly_rate,
        "JobInvolvement": job_involvement,
        "JobLevel": job_level,
        "JobRole": job_role,
        "JobSatisfaction": job_satisfaction,
        "MaritalStatus": marital_status,
        "MonthlyIncome": monthly_income,
        "MonthlyRate": monthly_rate,
        "NumCompaniesWorked": num_companies_worked,
        "OverTime": overtime,
        "PercentSalaryHike": percent_salary_hike,
        "PerformanceRating": performance_rating,
        "RelationshipSatisfaction": relationship_satisfaction,
        "StockOptionLevel": stock_option_level,
        "TotalWorkingYears": total_working_years,
        "TrainingTimesLastYear": training_times_last_year,
        "WorkLifeBalance": work_life_balance,
        "YearsAtCompany": years_at_company,
        "YearsInCurrentRole": years_in_current_role,
        "YearsSinceLastPromotion": years_since_last_promotion,
        "YearsWithCurrManager": years_with_curr_manager,
    }

    API_URL = "http://127.0.0.1:8000/predict"

    try:
        response = requests.post(
            API_URL,
            json=employee_data,
            timeout=30,
        )

        response.raise_for_status()
        result = response.json()

        
        st.divider()
        st.header("Prediction Result")

        probability = result["attrition_probability"]
        prediction = result["attrition_prediction"]
        threshold = result["classification_threshold"]

        # Risk categorization
        if probability < 0.10:
            risk_level = "Low"
            risk_color = "green"
        elif probability < threshold:
            risk_level = "Moderate"
            risk_color = "orange"
        else:
            risk_level = "High"
            risk_color = "red"

        # Prediction summary
        with st.container(border=True):
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Prediction",
                    "Likely to Leave" if prediction == "Yes" else "Likely to Stay"
                )

            with col2:
                st.metric(
                    "Attrition Probability",
                    f"{probability:.2%}"
                )

            with col3:
                st.metric("Risk Level", risk_level)

            st.progress(probability)

            if prediction == "Yes":
                st.error("Employee is predicted to leave.")
            else:
                st.success("Employee is predicted to stay.")

            st.caption(
                f"Classification threshold: {threshold:.0%} | "
                f"Model: {result['model_name']}"
            )
            
            
        # Risk interpretation
        st.subheader("Risk Interpretation")

        if risk_level == "Low":
            st.success(
                "Low Risk: The model estimates a relatively low "
                "probability of employee attrition."
            )
        elif risk_level == "Moderate":
            st.warning(
                "Moderate Risk: The estimated attrition probability "
                "is elevated. Consider reviewing relevant "
                "employee engagement factors."
            )
        else:
            st.error(
                "High Risk: The estimated attrition probability "
                "exceeds the model's classification threshold."
            )

        st.caption(
            "This prediction is an analytical estimate, not a "
            "certainty. Use it alongside human judgment and "
            "employee feedback."
        )


        # Employee profile summary
        with st.expander("View Employee Profile"):
            profile_col1, profile_col2, profile_col3 = st.columns(3)

            with profile_col1:
                st.write(f"**Age:** {age}")
                st.write(f"**Gender:** {gender}")
                st.write(f"**Marital Status:** {marital_status}")

            with profile_col2:
                st.write(f"**Department:** {department}")
                st.write(f"**Job Role:** {job_role}")
                st.write(f"**Job Level:** {job_level}")

            with profile_col3:
                st.write(f"**Monthly Income:** ₹{monthly_income:,}")
                st.write(f"**OverTime:** {overtime}")
                st.write(f"**Years at Company:** {years_at_company}")

    except requests.exceptions.RequestException as exc:
        st.error(f"API request failed: {exc}")
