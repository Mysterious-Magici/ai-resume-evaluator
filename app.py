import os
import json
import streamlit as st
import PyPDF2
from dotenv import load_dotenv
from google import genai
from pydantic import BaseModel

st.set_page_config(page_title="AI Resume Evaluator & Coach", layout="wide")

st.markdown("""
<style>
    /* Custom Styling for Skill Badges */
    .skill-badge-match {
        background-color: #DCFCE7; color: #166534;
        padding: 6px 14px; border-radius: 999px;
        font-size: 14px; font-weight: 600; display: inline-block;
        margin: 4px; border: 1px solid #BBF7D0;
    }
    .skill-badge-missing {
        background-color: #FEE2E2; color: #991B1B;
        padding: 6px 14px; border-radius: 999px;
        font-size: 14px; font-weight: 600; display: inline-block;
        margin: 4px; border: 1px solid #FECACA;
    }
    /* Main container styling */
    .block-container { padding-top: 2rem; }
    h1, h2, h3 { color: #1E293B; font-family: 'Inter', sans-serif; }
</style>
""", unsafe_allow_html=True)

load_dotenv()
my_api_key = os.getenv("GEMINI_API_KEY") 
model_name = os.getenv("GEMINI_MODEL", "gemini-3.8-flash") 

if not my_api_key:
    st.error("GEMINI_API_KEY environment variable is not set.")
    st.stop()

client = genai.Client(api_key=my_api_key)

class ResumeEvaluation(BaseModel):
    match_percentage: int
    matched_skills: list[str]
    missing_skills: list[str]
    brief_feedback: str

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "resume_text" not in st.session_state:
    st.session_state.resume_text = ""
if "job_description" not in st.session_state:
    st.session_state.job_description = ""

def extract_text_from_pdf(uploaded_file):
    reader = PyPDF2.PdfReader(uploaded_file)
    return "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])

def render_badges(skills, badge_type="match"):
    css_class = "skill-badge-match" if badge_type == "match" else "skill-badge-missing"
    html = "".join([f'<span class="{css_class}">{skill}</span>' for skill in skills])
    st.markdown(html, unsafe_allow_html=True)

st.title("AI Resume Evaluator & Career Coach")
st.markdown("Upload your resume, analyze your job match, and interact with your personal AI coach to improve your chances.")


tab1, tab2 = st.tabs(["Resume Match Analyzer", "AI Career Coach"])


with tab1:
    st.markdown("### Upload Details")
    col1, col2 = st.columns(2)
    
    with col1:
        job_desc_input = st.text_area("Paste the Job Description or Required Skills:", height=200, placeholder="e.g., Looking for a Python Developer with React and AWS experience...")
    with col2:
        uploaded_pdf = st.file_uploader("Upload your Resume (PDF format)", type=["pdf"])

    if st.button("Evaluate Match", type="primary", use_container_width=True):
        if not job_desc_input or not uploaded_pdf:
            st.warning(" Please provide both a Job Description and a Resume to proceed.")
        else:
            with st.spinner("AI is analyzing your profile..."):
                st.session_state.resume_text = extract_text_from_pdf(uploaded_pdf)
                st.session_state.job_description = job_desc_input
                
                system_prompt = """
                You are an expert ATS (Applicant Tracking System). 
                Compare the provided resume text against the job description.
                Extract exact matched skills, missing skills, match percentage (0-100), and a 1-sentence brief feedback.
                """
                user_prompt = f"JOB DESCRIPTION:\n{st.session_state.job_description}\n\nRESUME:\n{st.session_state.resume_text}"
                
                try:
                    response = client.models.generate_content(
                        model=model_name,
                        contents=user_prompt,
                        config={
                            "system_instruction": system_prompt,
                            "response_mime_type": "application/json",
                            "response_schema": ResumeEvaluation,
                            "temperature": 0.1
                        }
                    )
                    
                    evaluation = ResumeEvaluation(**json.loads(response.text))
                    
                    st.divider()
                    st.markdown("### Analysis Results")
                    
                
                    metric_col, feedback_col = st.columns([1, 3])
                    with metric_col:
                        st.metric(label="Match Score", value=f"{evaluation.match_percentage}%")
                    with feedback_col:
                        st.info(f"**AI Feedback:** {evaluation.brief_feedback}")
                    
                    st.progress(evaluation.match_percentage / 100)
                    st.write("") 
                    
                    res_col1, res_col2 = st.columns(2)
                    with res_col1:
                        st.markdown("#### Matched Skills")
                        if evaluation.matched_skills:
                            render_badges(evaluation.matched_skills, "match")
                        else:
                            st.write("No direct skills matched.")
                            
                    with res_col2:
                        st.markdown("#### Missing Skills")
                        if evaluation.missing_skills:
                            render_badges(evaluation.missing_skills, "missing")
                        else:
                            st.success("🎉 You have all the required skills!")
                            
                except Exception as e:
                    st.error(f"An error occurred: {e}")

with tab2:
    st.markdown("### Interact with your AI Career Coach")
    
    if not st.session_state.resume_text or not st.session_state.job_description:
        st.info(" Welcome! Please analyze your resume in the first tab to unlock your personalized AI Coach.")
    else:
        st.markdown("Your AI Coach knows your resume and the job description. Ask for advice, interview tips, or a cover letter!")
        
        st.markdown("##### Quick Questions")
        btn_col1, btn_col2, btn_col3 = st.columns(3)
        
        def set_prompt(text):
            st.session_state.active_prompt = text
            
        btn_col1.button("How to improve my score?", on_click=set_prompt, args=("How can I improve my match score for this specific role based on my current resume?",), use_container_width=True)
        btn_col2.button("Recommended Projects", on_click=set_prompt, args=("What specific side-projects should I build to learn the missing skills for this job?",), use_container_width=True)
        btn_col3.button("Draft Cover Letter", on_click=set_prompt, args=("Draft a short, professional cover letter for this role highlighting my matched skills.",), use_container_width=True)
        
        st.divider()

        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
        user_input = st.chat_input("Ask your AI Coach a question...")
        
        if "active_prompt" in st.session_state and st.session_state.active_prompt:
            user_input = st.session_state.active_prompt
            del st.session_state.active_prompt
            
        if user_input:
    
            with st.chat_message("user"):
                st.markdown(user_input)
            
            
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            
            
            system_instruction = f"""
            You are a supportive, expert AI Career Coach. 
            The user is applying for a job.
            Job Description:\n{st.session_state.job_description}\n
            User's Resume:\n{st.session_state.resume_text}\n
            Answer their questions highly professionally, concisely, and use bullet points where helpful.
            """
            
            
            messages_for_api = [{"role": "user" if m["role"] == "user" else "model", "parts": [{"text": m["content"]}]} for m in st.session_state.chat_history]
            
            with st.chat_message("assistant"):
                with st.spinner("Typing..."):
                    try:
                        chat_response = client.models.generate_content(
                            model="gemini-3.8-flash",
                            contents=messages_for_api,
                            config={"system_instruction": system_instruction, "temperature": 0.7}
                        )
                        bot_reply = chat_response.text
                        st.markdown(bot_reply)
                        
                        
                        st.session_state.chat_history.append({"role": "assistant", "content": bot_reply})
                    except Exception as e:
                        st.error(f"Chat Error: {e}")
