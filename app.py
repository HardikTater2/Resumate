import asyncio
import sys
import os
import base64
import pandas as pd
import streamlit as st

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["STREAMLIT_WATCHER_DISABLE_AUTO_WATCH"] = "1"

if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from core import enhanced_screen_resume

st.set_page_config(
    page_title="RESUMATE",
    page_icon="📄",
    layout="wide"
)

# --- Custom CSS for modern look ---
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .stButton>button {
        background: linear-gradient(45deg, #4CAF50, #2E7D32);
        color: white;
        border-radius: 8px;
        padding: 8px 16px;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    [role="tab"] {
        padding: 12px 24px !important;
        border-radius: 8px 8px 0 0 !important;
        background-color: #e9ecef !important;
        transition: all 0.3s;
    }
    [role="tab"][aria-selected="true"] {
        background: linear-gradient(45deg, #2196F3, #21CBF3) !important;
        color: white !important;
        font-weight: bold;
    }
    .dataframe {
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .card {
        background: white;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        border-left: 4px solid #2196F3;
    }
    .stSpinner > div {
        border-top-color: #2196F3 !important;
    }
</style>
""", unsafe_allow_html=True)

# --- Helper for PDF viewer ---
def show_pdf(file) -> None:
    if file is None:
        st.info("Upload a PDF to view it here.")
        return
    file.seek(0)
    file_bytes = file.read()
    base64_pdf = base64.b64encode(file_bytes).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="800px" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

# --- Tabs: Home, Viewer, Summary, Scores ---
tab_home, tab_viewer, tab_summary, tab_scores = st.tabs(
    ["🏠 Start Here", "📂 Document Viewer", "📝 Analysis Summary", "📊 Scores Dashboard"]
)

# --- State to store results ---
if "results_dict" not in st.session_state:
    st.session_state["results_dict"] = {}
if "submitted" not in st.session_state:
    st.session_state["submitted"] = False

# --- Tab: Home (Main Input Flow) ---
with tab_home:
    st.title("📄 RESUMATE")
    st.markdown("Upload your resumes and enter the job description below. Click **Analyze Resumes** to start screening.")
    st.markdown("---")
    resume_files = st.file_uploader(
        "Select PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload one or more PDF resumes."
    )
    job_description = st.text_area(
        "Paste job description here...",
        height=200,
        help="Enter the job description to match against resumes."
    )
    submit = st.button(
        "Analyze Resumes",
        type="primary"
    )

    if submit:
        st.session_state["results_dict"] = {}
        st.session_state["submitted"] = False
        if not resume_files:
            st.error("❗ Please upload at least one resume PDF")
        elif not job_description.strip():
            st.error("❗ Please enter a job description")
        else:
            progress_bar = st.progress(0)
            status_text = st.empty()
            for i, resume_file in enumerate(resume_files):
                status_text.info(f"🔍 Analyzing {i+1}/{len(resume_files)}: {resume_file.name}")
                progress_bar.progress((i+1)/len(resume_files))
                temp_filename = f"temp_{resume_file.name}"
                with open(temp_filename, "wb") as f:
                    f.write(resume_file.getbuffer())
   
                    results = enhanced_screen_resume(temp_filename, job_description)
                    st.session_state["results_dict"][resume_file.name] = {
                        "results": results,
                        "file": resume_file
                    }
                os.remove(temp_filename)
            progress_bar.empty()
            status_text.success("✅ Processing complete!")
            st.session_state["submitted"] = True
            st.balloons()
            st.snow()
            st.success("You can now view results in the other tabs!")

# --- Tab: Viewer ---
with tab_viewer:
    st.header("Document Viewer")
    if st.session_state.get("submitted") and st.session_state["results_dict"]:
        for filename, data_dict in st.session_state["results_dict"].items():
            with st.expander(f"📄 {filename}", expanded=False):
                show_pdf(data_dict["file"])
    else:
        st.info("Go to 'Start Here', upload resumes, and analyze to view documents.", icon="ℹ️")

# --- Tab: Summary ---
with tab_summary:
    st.header("Analysis Summary")
    if st.session_state.get("submitted") and st.session_state["results_dict"]:
        for filename, data_dict in st.session_state["results_dict"].items():
            with st.expander(f"📝 {filename}", expanded=True):
                results = data_dict["results"]
                resume_analysis = results.get("resume_analysis", {})
                jd_analysis = results.get("jd_analysis", {})
                st.subheader("Resume Breakdown")
                for key, value in resume_analysis.items():
                    section_title = key.replace("template_", "").replace("_", " ").upper()
                    with st.container():
                        st.markdown(f"""
                        <div class="card">
                            <h3>{section_title}</h3>
                            <div style="margin-top:12px">
                        """, unsafe_allow_html=True)
                        if value:
                            st.write(value)
                        else:
                            st.info("No data found for this section")
                        st.markdown("</div></div>", unsafe_allow_html=True)
    else:
        st.info("Go to 'Start Here', upload resumes, and analyze to view the summary.", icon="ℹ️")

# --- Tab: Scores Table ---
with tab_scores:
    st.header("Similarity Scores")
    if st.session_state.get("submitted") and st.session_state["results_dict"]:
        # Add threshold input
        threshold = st.slider(
            "Recommendation Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.05,
            help="Set the minimum overall score for recommendation"
        )
        
        data = []
        for filename, data_dict in st.session_state["results_dict"].items():
            scores = data_dict["results"].get("scores", {})
            overall_score = scores.get("overall", 0)
            row = {
                "File": filename,
                "Overall": overall_score,
                "Skills": scores.get("skills", 0),
                "Experience": scores.get("experience", 0),
                "Recommended": "✅ Yes" if overall_score >= threshold else "❌ No"
            }
            data.append(row)
        
        df_scores = pd.DataFrame(data)
        
        
        
        # Enhanced table with conditional formatting
        st.subheader("Detailed Scores")
        formatted_df = df_scores.copy()
        formatted_df[['Overall', 'Skills', 'Experience']] = formatted_df[['Overall', 'Skills', 'Experience']].applymap(
            lambda x: f"{x:.2f}" if isinstance(x, (int, float)) else x
        )
        
        # Apply conditional formatting to Recommended column
        def color_recommended(val):
            color = 'green' if val == "✅ Yes" else 'red'
            return f'color: {color}; font-weight: bold'
        
        st.dataframe(
            formatted_df.style
            .set_properties(**{'text-align': 'center'})
            .map(color_recommended, subset=['Recommended'])
            .set_table_styles([{'selector': 'th', 'props': [('text-align', 'center')]}]),
            use_container_width=True
        )
        
        # Summary statistics
        recommended_count = (df_scores['Overall'] >= threshold).sum()
        st.metric("Recommended Candidates", f"{recommended_count}/{len(df_scores)}")
        
    else:
        st.info("Go to 'Start Here', upload resumes, and analyze to view similarity scores.", icon="ℹ️")
