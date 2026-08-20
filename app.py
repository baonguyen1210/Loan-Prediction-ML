import gradio as gr
import pandas as pd
import joblib


# =========================
# 1. LOAD MODEL
# =========================

model = joblib.load("loan_model.pkl")


# =========================
# 2. SUBGRADE ENCODING
# =========================

subgrade_mapping = {
    "A1": 0,
    "A2": 1,
    "A3": 2,
    "A4": 3,
    "A5": 4,
    "B1": 5,
    "B2": 6,
    "B3": 7,
    "B4": 8,
    "B5": 9,
    "C1": 10,
    "C2": 11,
    "C3": 12,
    "C4": 13,
    "C5": 14,
    "D1": 15,
    "D2": 16,
    "D3": 17,
    "D4": 18,
    "D5": 19,
    "E1": 20,
    "E2": 21,
    "E3": 22,
    "E4": 23,
    "E5": 24,
    "F1": 25,
    "F2": 26,
    "F3": 27,
    "F4": 28,
    "F5": 29,
    "G1": 30,
    "G2": 31,
    "G3": 32,
    "G4": 33,
    "G5": 34
}


# =========================
# 3. PREDICTION FUNCTION
# =========================

def predict_loan(
    term,
    subGrade,
    annualIncome,
    verificationStatus,
    dti,
    ficoRangeLow,
    ficoRangeHigh,
    revolUtil,
    issueDate_months,
    debtpca1_1,
    npca1_1,
    npca1_2,
    netprofit
):

    # Convert SubGrade from text to number
    subGrade_encoded = subgrade_mapping[subGrade]

    # Create input DataFrame
    input_data = pd.DataFrame([{
        "term": term,
        "subGrade": subGrade_encoded,
        "annualIncome": annualIncome,
        "verificationStatus": verificationStatus,
        "dti": dti,
        "ficoRangeLow": ficoRangeLow,
        "ficoRangeHigh": ficoRangeHigh,
        "revolUtil": revolUtil,
        "issueDate_months": issueDate_months,
        "debtpca1-1": debtpca1_1,
        "npca1-1": npca1_1,
        "npca1-2": npca1_2,
        "netprofit": netprofit
    }])

    # Model prediction
    probability = model.predict_proba(input_data)[0][1]

    # Calculate repayment probability
    repayment_probability = 1 - probability

    # Risk classification
    if probability < 0.30:

        risk_level = "🟢 KHẢ NĂNG TRẢ NỢ CAO"

        explanation = (
            "Mô hình đánh giá khoản vay này có khả năng trả nợ "
            "cao hơn đáng kể so với khả năng vỡ nợ."
        )

    elif probability < 0.50:

        risk_level = "🟡 RỦI RO TRUNG BÌNH"

        explanation = (
            "Mô hình đánh giá khoản vay này có mức rủi ro trung bình. "
            "Nên xem xét thêm các thông tin tài chính trước khi quyết định."
        )

    else:

        risk_level = "🔴 RỦI RO CAO"

        explanation = (
            "Mô hình đánh giá khoản vay này có khả năng vỡ nợ cao. "
            "Nên xem xét kỹ trước khi phê duyệt."
        )

    # Return result to Gradio
    return (
        f"## {risk_level}\n\n"
        f"**Khả năng trả nợ:** {repayment_probability:.2%}\n\n"
        f"**Khả năng vỡ nợ:** {probability:.2%}\n\n"
        f"**Diễn giải:** {explanation}"
    )


# =========================
# 4. GRADIO INTERFACE
# =========================

demo = gr.Interface(

    fn=predict_loan,

    inputs=[

        gr.Radio(
            choices=[3, 5],
            label="Term",
            value=3
        ),

        gr.Dropdown(
            choices=list(subgrade_mapping.keys()),
            label="Sub Grade",
            value="A1"
        ),

        gr.Number(
            label="Annual Income",
            value=50000
        ),

        gr.Dropdown(
            choices=[0, 1, 2],
            label="Verification Status",
            value=0
        ),

        gr.Number(
            label="DTI",
            value=10
        ),

        gr.Number(
            label="FICO Range Low",
            value=700
        ),

        gr.Number(
            label="FICO Range High",
            value=704
        ),

        gr.Number(
            label="Revolving Utilization",
            value=30
        ),

        gr.Number(
            label="Issue Date (Months)",
            value=0
        ),

        gr.Number(
            label="Debt PCA1-1",
            value=0
        ),

        gr.Number(
            label="NPCA1-1",
            value=0
        ),

        gr.Number(
            label="NPCA1-2",
            value=0
        ),

        gr.Number(
            label="Net Profit",
            value=0
        )
    ],

    outputs=gr.Markdown(
        label="Kết quả dự báo"
    ),

    title="Loan Default Prediction",

    description=(
        "Ứng dụng Machine Learning dự đoán khả năng "
        "trả nợ và vỡ nợ của khoản vay."
    )
)


# =========================
# 5. LAUNCH APP
# =========================

import os

demo.launch(
    server_name="0.0.0.0",
    server_port=int(os.environ.get("PORT", 7860))
)