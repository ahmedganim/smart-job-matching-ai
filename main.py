import os
import io
import re
import traceback
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from supabase import create_client, Client

SUPABASE_URL = "https://kkjlyqmhxcrzfcwzkiin.supabase.co"
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imtramx5cW1oeGNyemZjd3praWluIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc5NDIzMTQzNSwiZXhwIjoyMTA5ODA3NDM1fQ.u-E76MrP-o9ns9fPL4di9VTgnk_lNWyijLchHyNa0Vs")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

app = FastAPI(title="Smart Job Matching AI Backend")

class ProcessCvRequest(BaseModel):
    application_id: str

@app.get("/")
def health_check():
    return {"status": "online", "engine": "scikit-learn-lightweight"}

def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    full_text = ""
    for page in reader.pages:
        text = page.extract_text()
        if text:
            full_text += text + "\n"
    return full_text

def extract_experience_years(text: str) -> float:
    match = re.search(r'(\d+)\+?\s*(?:years?|yrs?)\s*(?:of)?\s*experience', text, re.IGNORECASE)
    if match:
        return float(match.group(1))
    years = re.findall(r'\b(20\d{2})\b', text)
    if len(years) >= 2:
        diff = max([int(y) for y in years]) - min([int(y) for y in years])
        if 0 < diff <= 30:
            return float(diff)
    return 1.0

def calculate_skills_overlap(cv_text: str, required_skills: list) -> float:
    if not required_skills:
        return 1.0
    matched = 0
    cv_text_lower = cv_text.lower()
    for skill in required_skills:
        pattern = r'\b' + re.escape(skill.strip().lower()) + r'\b'
        if re.search(pattern, cv_text_lower):
            matched += 1
    return matched / len(required_skills)

def calculate_semantic_similarity(text1: str, text2: str) -> float:
    if not text1.strip() or not text2.strip():
        return 0.5
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform([text1, text2])
    sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    return float(max(0.0, min(1.0, sim)))

@app.post("/process-cv")
async def process_cv(request: ProcessCvRequest):
    print(f"--- Processing Application: {request.application_id} ---")
    try:
        app_res = supabase.table("applications").select("*").eq("id", request.application_id).single().execute()
        app_data = app_res.data
        if not app_data:
            print("Error: Application not found")
            raise HTTPException(status_code=404, detail="Application not found")

        job_id = app_data.get("job_id")
        cv_id = app_data.get("cv_id")

        job_res = supabase.table("jobs").select("*").eq("id", job_id).single().execute()
        job = job_res.data or {}

        cv_res = supabase.table("cvs").select("*").eq("id", cv_id).single().execute()
        cv = cv_res.data or {}

        file_path = cv.get("file_path") or cv.get("url") or cv.get("pdf_url")
        if not file_path:
            raise HTTPException(status_code=400, detail="CV file_path is missing in cvs table")

        if "resumes/" in file_path:
            file_path = file_path.split("resumes/")[-1]

        print(f"Downloading file: {file_path}")
        pdf_bytes = supabase.storage.from_("resumes").download(file_path)
        cv_text = extract_text_from_pdf(pdf_bytes)

        job_title = job.get("title", "")
        job_desc = job.get("description", "")
        job_context = f"{job_title} {job_desc}"

        semantic_score = calculate_semantic_similarity(cv_text, job_context)
        
        required_skills = job.get("required_skills") or []
        if isinstance(required_skills, str):
            required_skills = [s.strip() for s in required_skills.split(",")]
            
        skills_score = calculate_skills_overlap(cv_text, required_skills)
        detected_experience = extract_experience_years(cv_text)
        required_exp = float(job.get("min_experience") or 1.0)
        exp_score = min(1.0, detected_experience / required_exp) if required_exp > 0 else 1.0

        final_match_score = round(((semantic_score * 0.50) + (skills_score * 0.35) + (exp_score * 0.15)) * 100, 1)
        detected_skills = [s for s in required_skills if s.lower() in cv_text.lower()]
        summary_snippet = cv_text.strip().replace("\n", " ")[:200] + "..."

        supabase.table("cvs").update({
            "summary": summary_snippet,
            "skills": detected_skills,
            "experience_years": detected_experience,
            "raw_text": cv_text
        }).eq("id", cv_id).execute()

        supabase.table("applications").update({
            "match_score": final_match_score
        }).eq("id", request.application_id).execute()

        print(f"Success! Score: {final_match_score}, Exp: {detected_experience}")
        return {
            "status": "success",
            "match_score": final_match_score,
            "detected_experience": detected_experience,
            "matched_skills": detected_skills
        }
    except Exception as e:
        print(f"EXCEPTION IN PROCESS-CV: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))
