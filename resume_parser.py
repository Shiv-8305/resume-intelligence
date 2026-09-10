"""
Multi-format resume parser supporting PDF and image files with deterministic and LLM-assisted extraction.
"""
import re
import uuid
import logging
from typing import Tuple, List, Dict, Any, Optional
import pymupdf as fitz
from PIL import Image
import io

from models import CandidateProfile, WorkExperience, EducationItem
from config import OPENAI_API_KEY, DEFAULT_LLM_MODEL, JOB_ROLE_TAXONOMY

logger = logging.getLogger(__name__)

# Master list of common technical skills for regex heuristic matching
COMMON_SKILLS = set([
    "python", "java", "c++", "c#", "javascript", "typescript", "html", "css", "sql", "nosql",
    "fastapi", "django", "flask", "spring", "spring boot", "react", "next.js", "angular", "vue",
    "node.js", "express", "postgresql", "mysql", "mongodb", "redis", "dynamodb", "aws", "azure",
    "gcp", "docker", "kubernetes", "git", "github", "gitlab", "ci/cd", "linux", "rest api",
    "graphql", "microservices", "machine learning", "deep learning", "nlp", "computer vision",
    "pytorch", "tensorflow", "scikit-learn", "pandas", "numpy", "tableau", "powerbi", "excel",
    "statistics", "spark", "hadoop", "airflow", "kafka", "snowflake", "bigquery", "etl",
    "generative ai", "llm", "langchain", "llamaindex", "rag", "vector db", "prompt engineering",
    "system design", "algorithms", "data structures", "agile", "jira", "unit testing", "pytest"
])


def extract_text_from_pdf(pdf_bytes: bytes) -> Tuple[str, bool]:
    """
    Extracts text from PDF bytes using PyMuPDF.
    Returns (extracted_text, is_scanned_flag).
    """
    extracted_text = ""
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        for page in doc:
            text = page.get_text("text")
            if text:
                extracted_text += text + "\n"
        doc.close()
    except Exception as e:
        logger.error(f"PyMuPDF PDF extraction failed: {e}")

    extracted_text = extracted_text.strip()
    word_count = len(extracted_text.split())
    is_scanned = word_count < 30  # Threshold for scanned/image-based PDFs

    return extracted_text, is_scanned


def extract_text_from_image(image_bytes: bytes) -> Tuple[str, bool]:
    """
    Extracts text from image bytes (PNG, JPG, JPEG, WEBP) using PyTesseract OCR.
    """
    extracted_text = ""
    try:
        image = Image.open(io.BytesIO(image_bytes))
        import pytesseract
        extracted_text = pytesseract.image_to_string(image)
    except Exception as e:
        logger.warning(f"PyTesseract OCR image extraction failed or not installed: {e}")
        extracted_text = ""

    extracted_text = extracted_text.strip()
    is_scanned = len(extracted_text.split()) < 15
    return extracted_text, is_scanned


def extract_email(text: str) -> str:
    match = re.search(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', text)
    return match.group(0) if match else ""


def extract_phone(text: str) -> str:
    # Match standard phone numbers with or without country codes
    pattern = r'(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'
    match = re.search(pattern, text)
    return match.group(0) if match else ""


def extract_name(text: str) -> str:
    """Extract candidate name from header lines."""
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    for line in lines[:5]:
        # Ignore lines with email, phone, or URLs
        if '@' in line or 'http' in line or re.search(r'\d{5,}', line):
            continue
        # Names are usually 2 to 4 words
        words = line.split()
        if 1 <= len(words) <= 4 and all(w[0].isupper() for w in words if w[0].isalpha()):
            return line
    return "Candidate"


def extract_experience_years(text: str) -> float:
    """Extracts total years of experience using regex patterns and date ranges."""
    # Direct mention: "X+ years of experience"
    exp_matches = re.findall(r'(\d+(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?)\s*(?:of)?\s*(?:experience|exp)?', text, re.IGNORECASE)
    if exp_matches:
        try:
            years = [float(m) for m in exp_matches if float(m) <= 40]
            if years:
                return max(years)
        except ValueError:
            pass

    # Date range search: e.g. "2020 - 2024" or "2019 - Present"
    year_ranges = re.findall(r'(20\d{2})\s*[-–—to]+\s*(20\d{2}|present|current)', text, re.IGNORECASE)
    total_months = 0
    import datetime
    current_year = datetime.datetime.now().year

    for start_str, end_str in year_ranges:
        try:
            start_yr = int(start_str)
            end_yr = current_year if end_str.lower() in ["present", "current"] else int(end_str)
            if end_yr >= start_yr:
                total_months += (end_yr - start_yr) * 12
        except ValueError:
            pass

    if total_months > 0:
        return round(total_months / 12.0, 1)

    return 0.0


def extract_skills(text: str) -> List[str]:
    """Matches text against known technical skills."""
    found_skills = set()
    text_lower = text.lower()

    # Search common skills
    for skill in COMMON_SKILLS:
        pattern = r'\b' + re.escape(skill) + r'\b'
        if re.search(pattern, text_lower):
            # Capitalize nicely
            found_skills.add(skill.title() if len(skill) > 3 else skill.upper())

    # Search role taxonomy skills
    for role_info in JOB_ROLE_TAXONOMY.values():
        for skill in role_info["required_skills"] + role_info["preferred_skills"]:
            if re.search(r'\b' + re.escape(skill.lower()) + r'\b', text_lower):
                found_skills.add(skill)

    return sorted(list(found_skills))


def extract_education(text: str) -> List[EducationItem]:
    """Extracts education degrees and fields of study."""
    education_items = []
    degree_patterns = [
        r"(B\.?Tech|M\.?Tech|B\.?S|M\.?S|Ph\.?D|Bachelor|Master|Diploma|Associate)\s*(?:in|of)?\s*([A-Za-z\s]+)?",
        r"(Computer Science|Information Technology|Electrical Engineering|Data Science|Software Engineering)"
    ]

    for pattern in degree_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            degree_str = match.group(0).strip()
            if len(degree_str) > 3 and not any(e.degree == degree_str for e in education_items):
                education_items.append(EducationItem(degree=degree_str, field_of_study="Computer Science/STEM"))

    return education_items[:3]


def extract_job_titles(text: str) -> List[str]:
    """Extracts job titles mentioned in resume."""
    titles = set()
    for role, data in JOB_ROLE_TAXONOMY.items():
        for title_variant in data["title_variants"]:
            if re.search(r'\b' + re.escape(title_variant) + r'\b', text, re.IGNORECASE):
                titles.add(title_variant.title())
    return sorted(list(titles))


def parse_resume_deterministically(filename: str, raw_text: str, is_scanned: bool, file_type: str) -> CandidateProfile:
    """
    Deterministic rule-based parser that guarantees zero-crash candidate extraction.
    """
    cand_id = str(uuid.uuid4())[:8]
    name = extract_name(raw_text)
    email = extract_email(raw_text)
    phone = extract_phone(raw_text)
    exp_years = extract_experience_years(raw_text)
    skills = extract_skills(raw_text)
    education = extract_education(raw_text)
    job_titles = extract_job_titles(raw_text)

    # Headline generation
    headline = job_titles[0] if job_titles else "Technology Professional"

    return CandidateProfile(
        id=cand_id,
        filename=filename,
        file_type=file_type,
        name=name,
        email=email,
        phone=phone,
        location="Not Specified",
        headline=headline,
        job_titles=job_titles,
        experience_years=exp_years,
        skills=skills,
        education=education,
        work_experience=[
            WorkExperience(
                title=job_titles[0] if job_titles else "Software Specialist",
                description="Professional software development & engineering experience."
            )
        ] if job_titles else [],
        raw_text=raw_text,
        is_scanned=is_scanned,
        parsing_method="deterministic"
    )


def parse_resume_with_llm(raw_text: str) -> Optional[Dict[str, Any]]:
    """Attempts structured LLM resume parsing with OpenAI API if available."""
    if not OPENAI_API_KEY:
        return None

    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)

        prompt = f"""
Extract key resume details from the text below into strict JSON matching this structure:
{{
  "name": "Full Name",
  "email": "email@example.com",
  "phone": "+1...",
  "location": "City, Country",
  "headline": "Professional headline",
  "job_titles": ["Title 1", "Title 2"],
  "experience_years": 4.5,
  "skills": ["Skill1", "Skill2"],
  "education": [{"degree": "B.Tech", "institution": "University", "field_of_study": "Computer Science"}],
  "certifications": ["Cert 1"],
  "projects": ["Project 1"],
  "work_experience": [{"title": "Dev", "company": "Co", "duration": "2020-2023", "description": "built APIs"}],
  "achievements": [],
  "languages": ["English"]
}}

Resume Text:
{raw_text[:4000]}
"""
        response = client.chat.completions.create(
            model=DEFAULT_LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.1
        )
        import json
        data = json.loads(response.choices[0].message.content)
        return data
    except Exception as e:
        logger.warning(f"LLM Resume Parsing failed, falling back to deterministic: {e}")
        return None


def parse_resume_file(filename: str, file_bytes: bytes) -> CandidateProfile:
    """
    Main entrypoint to parse PDF or Image resumes.
    Automatically handles layout extraction, scanned detection, deterministic & LLM extraction.
    """
    is_image = filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))
    file_type = "image" if is_image else "pdf"

    if is_image:
        raw_text, is_scanned = extract_text_from_image(file_bytes)
    else:
        raw_text, is_scanned = extract_text_from_pdf(file_bytes)

    # Attempt LLM parsing first if API key is active and text is present
    if not is_scanned and OPENAI_API_KEY:
        llm_data = parse_resume_with_llm(raw_text)
        if llm_data and isinstance(llm_data, dict):
            try:
                cand_id = str(uuid.uuid4())[:8]
                return CandidateProfile(
                    id=cand_id,
                    filename=filename,
                    file_type=file_type,
                    name=llm_data.get("name") or extract_name(raw_text),
                    email=llm_data.get("email") or extract_email(raw_text),
                    phone=llm_data.get("phone") or extract_phone(raw_text),
                    location=llm_data.get("location") or "Not Specified",
                    headline=llm_data.get("headline") or "Software Professional",
                    job_titles=llm_data.get("job_titles") or extract_job_titles(raw_text),
                    experience_years=float(llm_data.get("experience_years") or extract_experience_years(raw_text)),
                    skills=llm_data.get("skills") or extract_skills(raw_text),
                    education=[EducationItem(**e) for e in llm_data.get("education", []) if isinstance(e, dict)] or extract_education(raw_text),
                    certifications=llm_data.get("certifications") or [],
                    projects=llm_data.get("projects") or [],
                    work_experience=[WorkExperience(**w) for w in llm_data.get("work_experience", []) if isinstance(w, dict)],
                    achievements=llm_data.get("achievements") or [],
                    languages=llm_data.get("languages") or [],
                    raw_text=raw_text,
                    is_scanned=is_scanned,
                    parsing_method="llm"
                )
            except Exception as e:
                logger.warning(f"Error instantiating CandidateProfile from LLM data: {e}")

    # Fallback: Deterministic Extraction
    return parse_resume_deterministically(filename, raw_text, is_scanned, file_type)
