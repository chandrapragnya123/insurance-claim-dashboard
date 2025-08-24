import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

# Load dataset
df = pd.read_csv("synthetic_claims_dataset.csv")

st.set_page_config(page_title="Insurance Claim Dashboard", layout="wide")

st.title("🏥 Insurance Claim Insights")
st.markdown("Easily check claim denials, reasons, and fixes.")

# --- Sidebar filters ---
st.sidebar.header("🔍 Search & Filter")
search_type = st.sidebar.radio("Search by:", ["Patient ID", "Payer Type", "State Analysis", "Denial Reasons"])

# Convert object columns to string
for col in df.select_dtypes('object').columns:
    df[col] = df[col].astype(str)

# ---------------- Patient ID Search ----------------
if search_type == "Patient ID":
    patient_id = st.sidebar.text_input("Enter Patient ID (e.g., P1456)")
    if patient_id:
        patient_data = df[df["Patient_ID"] == patient_id]

        if not patient_data.empty:
            st.subheader(f"Details for Patient ID: {patient_id}")
            st.dataframe(patient_data)

            total_claims = patient_data.shape[0]
            denied = patient_data["Denied"].sum()
            accepted = total_claims - denied

            st.write(f"**Total Claims:** {total_claims}")
            st.write(f"✅ Accepted Claims: {accepted} ({(accepted/total_claims)*100:.2f}%)")
            st.write(f"❌ Denied Claims: {denied} ({(denied/total_claims)*100:.2f}%)")

            if denied > 0:
                st.error("Some claims were denied.")
                st.write("### Denial Reasons & Recommended Fixes:")
                st.dataframe(patient_data[["Denial_Reason", "Recommended_Fix"]].dropna())
            else:
                st.success("All claims accepted ✅")
        else:
            st.warning("No records found for that Patient ID.")

# ---------------- Payer Type Search ----------------
elif search_type == "Payer Type":
    payer_type = st.sidebar.text_input("Enter Payer Type (e.g., Medicare, Medicaid, Private)")
    if payer_type:
        payer_data = df[df["Payer_Type"].str.lower() == payer_type.lower()]

        if not payer_data.empty:
            st.subheader(f"Details for Payer Type: {payer_type}")

            total_claims = payer_data.shape[0]
            denied = payer_data["Denied"].sum()
            accepted = total_claims - denied

            st.write(f"**Total Claims:** {total_claims}")
            st.write(f"✅ Accepted Claims: {accepted} ({(accepted/total_claims)*100:.2f}%)")
            st.write(f"❌ Denied Claims: {denied} ({(denied/total_claims)*100:.2f}%)")

            # --- Top 5 Payers by denial rate ---
            top_groups = df.groupby("Payer_Type").agg(
                total_claims=("Denied", "count"),
                denied_claims=("Denied", "sum")
            )
            top_groups["denial_rate"] = (top_groups["denied_claims"] / top_groups["total_claims"]) * 100
            top5 = top_groups.sort_values("denial_rate", ascending=False).head(5)

            st.write("### Top 5 Payers by Denial Rate")
            st.bar_chart(top5["denial_rate"])

        else:
            st.warning("No records found for that Payer Type.")

# ---------------- State-Level Heatmap ----------------
elif search_type == "State Analysis":
    st.subheader("📍 Claim Denial Rates by State")

    state_summary = df.groupby("Patient_State").agg(
        total_claims=("Denied", "count"),
        denied_claims=("Denied", "sum")
    )
    state_summary["denial_rate"] = (state_summary["denied_claims"] / state_summary["total_claims"]) * 100
    state_summary = state_summary.reset_index()

    fig = px.choropleth(
        state_summary,
        locations="Patient_State",
        locationmode="USA-states",
        color="denial_rate",
        scope="usa",
        color_continuous_scale="Reds",
        labels={"denial_rate": "Denial Rate (%)"}
    )
    st.plotly_chart(fig, use_container_width=True)

# ---------------- Denial Reason Dashboard ----------------
elif search_type == "Denial Reasons":
    st.subheader("❌ Top Denial Reasons Across All Claims")

    reason_counts = df["Denial_Reason"].value_counts().head(10)
    fig, ax = plt.subplots()
    sns.barplot(x=reason_counts.values, y=reason_counts.index, ax=ax, palette="Reds_r")
    ax.set_xlabel("Number of Denials")
    ax.set_ylabel("Denial Reason")
    st.pyplot(fig)

# --- Export to PDF ---
if st.sidebar.button("📥 Download Summary Report"):
    doc = SimpleDocTemplate("claim_report.pdf")
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("Insurance Claim Summary Report", styles["Title"]))
    story.append(Spacer(1, 20))

    total_claims = df.shape[0]
    denied = df["Denied"].sum()
    accepted = total_claims - denied

    story.append(Paragraph(f"Total Claims: {total_claims}", styles["Normal"]))
    story.append(Paragraph(f"Accepted Claims: {accepted}", styles["Normal"]))
    story.append(Paragraph(f"Denied Claims: {denied}", styles["Normal"]))

    story.append(Spacer(1, 20))
    story.append(Paragraph("This report summarizes overall claim insights for decision making.", styles["Italic"]))

    doc.build(story)
    with open("claim_report.pdf", "rb") as f:
        st.download_button("Download PDF", f, file_name="claim_report.pdf")
# --- Dataset Overview ---
st.markdown("### 📊 Dataset Overview")

# Show basic info
st.write(f"**Total Rows:** {df.shape[0]}")
st.write(f"**Total Columns:** {df.shape[1]}")

# Show column names
st.write("**Columns:**", list(df.columns))

# Show missing values
missing = df.isnull().sum()
st.write("**Missing Values:**")
st.dataframe(missing[missing > 0])

# Show numerical statistics
st.markdown("#### Numerical Columns Summary")
st.dataframe(df.describe())

# Show categorical statistics
st.markdown("#### Categorical Columns Summary")
st.dataframe(df.describe(include='object'))
