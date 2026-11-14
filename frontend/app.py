
"""WorkforceIQ Streamlit application."""

import os

import plotly.graph_objects as go
import requests
import streamlit as st


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

API_URL = os.getenv(
    "API_URL",
    "https://workforceiq-api-prod.onrender.com/predict",
)

st.set_page_config(
    page_title="WorkforceIQ",
    page_icon="📊",
    layout="wide",
)


# ---------------------------------------------------------
# Live Prediction Charts
# ---------------------------------------------------------

def render_prediction_charts(probability, threshold):
    """Render live attrition probability charts."""

    # Risk classification
    if probability < 0.20:
        risk_category = "Low"
        risk_color = "#22c55e"
    elif probability < 0.50:
        risk_category = "Medium"
        risk_color = "#f59e0b"
    else:
        risk_category = "High"
        risk_color = "#ef4444"

    # Risk category
    st.subheader("Employee Attrition Risk")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Attrition Probability",
            f"{probability:.2%}",
        )

        st.metric(
            "Classification Threshold",
            f"{threshold:.2%}",
        )

        st.markdown(
            f"""
            <div style="
                background-color:{risk_color};
                padding:12px;
                border-radius:10px;
                text-align:center;
                color:white;
                font-weight:bold;
                font-size:20px;
            ">
                {risk_category} Risk
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        # Attrition probability gauge
        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=probability * 100,
                number={"suffix": "%"},
                title={"text": "Attrition Risk"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": risk_color},
                    "steps": [
                        {
                            "range": [0, 20],
                            "color": "#dcfce7",
                        },
                        {
                            "range": [20, 50],
                            "color": "#fef3c7",
                        },
                        {
                            "range": [50, 100],
                            "color": "#fee2e2",
                        },
                    ],
                },
            )
        )

        gauge.update_layout(
            template="plotly_dark",
            height=300,
            margin=dict(l=20, r=20, t=50, b=20),
        )

        st.plotly_chart(
            gauge,
            use_container_width=True,
        )

    # Stay vs Leave probability
    st.subheader("Stay vs Leave Probability")

    stay_probability = 1 - probability

    donut = go.Figure(
        go.Pie(
            labels=["Stay", "Leave"],
            values=[
                stay_probability,
                probability,
            ],
            hole=0.65,
            marker={
                "colors": [
                    "#22c55e",
                    "#ef4444",
                ]
            },
            textinfo="label+percent",
        )
    )

    donut.update_layout(
        template="plotly_dark",
        height=350,
        showlegend=True,
        margin=dict(l=20, r=20, t=30, b=20),
    )

    st.plotly_chart(
        donut,
        use_container_width=True,
    )


# ---------------------------------------------------------
# SHAP Explainability Chart
# ---------------------------------------------------------

def render_shap_chart(feature_contributions):
    """Render employee-specific SHAP feature contributions."""

    if not feature_contributions:
        st.info(
            "SHAP explanations are not available "
            "for this prediction."
        )
        return

    # Sort by absolute impact
    contributions = sorted(
        feature_contributions,
        key=lambda item: abs(item["impact"]),
    )

    # Clean feature names
    features = [
        item["feature"]
        .replace("num__", "")
        .replace("nom__", "")
        .replace("ord__", "")
        for item in contributions
    ]

    impacts = [
        item["impact"]
        for item in contributions
    ]

    # Positive SHAP increases attrition risk
    colors = [
        "#ef4444" if impact > 0 else "#22c55e"
        for impact in impacts
    ]

    fig = go.Figure(
        go.Bar(
            x=impacts,
            y=features,
            orientation="h",
            marker_color=colors,
            hovertemplate=(
                "<b>%{y}</b><br>"
                "SHAP impact: %{x:.4f}"
                "<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title="Top 10 Attrition Risk Drivers",
        xaxis_title="SHAP Impact on Attrition Probability",
        yaxis_title="Employee Features",
        template="plotly_dark",
        height=500,
        margin=dict(
            l=20,
            r=20,
            t=60,
            b=40,
        ),
        showlegend=False,
        xaxis=dict(
            zeroline=True,
            zerolinewidth=2,
        ),
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    st.caption(
        "Red: increases attrition risk | "
        "Green: decreases attrition risk"
    )


# ---------------------------------------------------------
# Application Header
# ---------------------------------------------------------

st.title("WorkforceIQ")
st.caption(
    "Employee Attrition Prediction & Analytics"
)

st.markdown(
    """
    Predict employee attrition probability and understand
    the factors influencing individual employee risk.
    """
)

st.divider()


# ---------------------------------------------------------
# Employee Input
# ---------------------------------------------------------

st.subheader("Employee Information")

with st.form("employee_prediction_form"):

    col1, col2, col3 = st.columns(3)

    with col1:
        Age = st.number_input(
            "Age",
            min_value=18,
            max_value=60,
            value=30,
        )

        BusinessTravel = st.selectbox(
            "Business Travel",
            [
                "Travel_Rarely",
                "Travel_Frequently",
                "Non-Travel",
            ],
        )

        DailyRate = st.number_input(
            "Daily Rate",
            min_value=1,
            value=1000,
        )

        Department = st.selectbox(
            "Department",
            [
                "Sales",
                "Research & Development",
                "Human Resources",
            ],
        )

        DistanceFromHome = st.number_input(
            "Distance From Home",
            min_value=1,
            value=10,
        )

        Education = st.selectbox(
            "Education",
            [1, 2, 3, 4, 5],
            index=2,
        )

        EducationField = st.selectbox(
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

    with col2:
        EnvironmentSatisfaction = st.selectbox(
            "Environment Satisfaction",
            [1, 2, 3, 4],
            index=2,
        )

        Gender = st.selectbox(
            "Gender",
            ["Male", "Female"],
        )

        HourlyRate = st.number_input(
            "Hourly Rate",
            min_value=1,
            value=65,
        )

        JobInvolvement = st.selectbox(
            "Job Involvement",
            [1, 2, 3, 4],
            index=2,
        )

        JobLevel = st.selectbox(
            "Job Level",
            [1, 2, 3, 4, 5],
            index=0,
        )

        JobRole = st.selectbox(
            "Job Role",
            [
                "Sales Executive",
                "Research Scientist",
                "Laboratory Technician",
                "Manufacturing Director",
                "Healthcare Representative",
                "Manager",
                "Sales Representative",
                "Research Director",
                "Human Resources",
            ],
        )

        JobSatisfaction = st.selectbox(
            "Job Satisfaction",
            [1, 2, 3, 4],
            index=2,
        )

    with col3:
        MaritalStatus = st.selectbox(
            "Marital Status",
            ["Single", "Married", "Divorced"],
        )

        MonthlyIncome = st.number_input(
            "Monthly Income",
            min_value=1,
            value=5000,
        )

        MonthlyRate = st.number_input(
            "Monthly Rate",
            min_value=1,
            value=15000,
        )

        NumCompaniesWorked = st.number_input(
            "Number of Companies Worked",
            min_value=0,
            value=2,
        )

        OverTime = st.selectbox(
            "OverTime",
            ["Yes", "No"],
        )

        PercentSalaryHike = st.number_input(
            "Percent Salary Hike",
            min_value=0,
            max_value=100,
            value=15,
        )

        PerformanceRating = st.selectbox(
            "Performance Rating",
            [3, 4],
        )

    # Remaining employee attributes
    col4, col5, col6 = st.columns(3)

    with col4:
        RelationshipSatisfaction = st.selectbox(
            "Relationship Satisfaction",
            [1, 2, 3, 4],
            index=2,
        )

        StockOptionLevel = st.selectbox(
            "Stock Option Level",
            [0, 1, 2, 3],
        )

        TotalWorkingYears = st.number_input(
            "Total Working Years",
            min_value=0,
            value=8,
        )

    with col5:
        TrainingTimesLastYear = st.number_input(
            "Training Times Last Year",
            min_value=0,
            value=2,
        )

        WorkLifeBalance = st.selectbox(
            "Work Life Balance",
            [1, 2, 3, 4],
            index=2,
        )

        YearsAtCompany = st.number_input(
            "Years At Company",
            min_value=0,
            value=5,
        )

    with col6:
        YearsInCurrentRole = st.number_input(
            "Years In Current Role",
            min_value=0,
            value=3,
        )

        YearsSinceLastPromotion = st.number_input(
            "Years Since Last Promotion",
            min_value=0,
            value=1,
        )

        YearsWithCurrManager = st.number_input(
            "Years With Current Manager",
            min_value=0,
            value=3,
        )

    submitted = st.form_submit_button(
        "Predict Attrition",
        use_container_width=True,
    )


# ---------------------------------------------------------
# Prediction
# ---------------------------------------------------------

if submitted:

    employee_data = {
        "Age": Age,
        "BusinessTravel": BusinessTravel,
        "DailyRate": DailyRate,
        "Department": Department,
        "DistanceFromHome": DistanceFromHome,
        "Education": Education,
        "EducationField": EducationField,
        "EnvironmentSatisfaction": EnvironmentSatisfaction,
        "Gender": Gender,
        "HourlyRate": HourlyRate,
        "JobInvolvement": JobInvolvement,
        "JobLevel": JobLevel,
        "JobRole": JobRole,
        "JobSatisfaction": JobSatisfaction,
        "MaritalStatus": MaritalStatus,
        "MonthlyIncome": MonthlyIncome,
        "MonthlyRate": MonthlyRate,
        "NumCompaniesWorked": NumCompaniesWorked,
        "OverTime": OverTime,
        "PercentSalaryHike": PercentSalaryHike,
        "PerformanceRating": PerformanceRating,
        "RelationshipSatisfaction": RelationshipSatisfaction,
        "StockOptionLevel": StockOptionLevel,
        "TotalWorkingYears": TotalWorkingYears,
        "TrainingTimesLastYear": TrainingTimesLastYear,
        "WorkLifeBalance": WorkLifeBalance,
        "YearsAtCompany": YearsAtCompany,
        "YearsInCurrentRole": YearsInCurrentRole,
        "YearsSinceLastPromotion": YearsSinceLastPromotion,
        "YearsWithCurrManager": YearsWithCurrManager,
    }

    try:

        with st.spinner(
            "Analyzing employee attrition risk..."
        ):

            response = requests.post(
                API_URL,
                json=employee_data,
                timeout=120,
            )

            response.raise_for_status()

            result = response.json()

        # Prediction results
        probability = result["attrition_probability"]
        prediction = result["attrition_prediction"]
        threshold = result["classification_threshold"]

        st.divider()
        st.subheader("Prediction Results")

        result_col1, result_col2 = st.columns(2)

        with result_col1:
            st.metric(
                "Predicted Attrition",
                prediction,
            )

        with result_col2:
            st.metric(
                "Attrition Probability",
                f"{probability:.2%}",
            )

        # Live prediction charts
        render_prediction_charts(
            probability=probability,
            threshold=threshold,
        )

        # Individual SHAP explainability
        st.divider()
        st.subheader(
            "Employee-Level Explainability"
        )

        render_shap_chart(
            result.get(
                "feature_contributions",
                [],
            )
        )

        st.success(
            "Employee prediction and explanation "
            "generated successfully."
        )

    except requests.exceptions.RequestException as exc:

        st.error(
            f"API request failed: {exc}"
        )

    except Exception as exc:

        st.error(
            f"Unexpected error: {exc}"
        )
