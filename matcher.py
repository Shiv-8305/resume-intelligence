"""
Job-Specific Matching Engine, ATS Scoring, Anonymization, and Tailored Question Generator.
"""
import re
import logging
from typing import List, Dict, Any, Optional, Tuple

from models import CandidateProfile, JobRequirement, MatchScoreBreakdown, CandidateMatchResult
from config import JOB_ROLE_TAXONOMY, DEFAULT_SCORE_WEIGHTS, OPENAI_API_KEY, DEFAULT_LLM_MODEL
from embeddings import SemanticSearchEngine

logger = logging.getLogger(__name__)


def parse_job_requirement(target_role: str, query: str = "", raw_jd: str = "", custom_taxonomy: Optional[Dict[str, Any]] = None) -> JobRequirement:
    """
    Constructs a structured JobRequirement by combining:
    1. Selected Role Taxonomy defaults (or custom taxonomy)
    2. Extracted Natural Language Query requirements
    3. Extracted Job Description requirements (via LLM or regex)
    """
    taxonomy = custom_taxonomy if custom_taxonomy else JOB_ROLE_TAXONOMY
    base_role_info = taxonomy.get(target_role, taxonomy.get("Software Engineer", JOB_ROLE_TAXONOMY["Software Engineer"]))

    req_skills = list(base_role_info.get("required_skills", []))
    pref_skills = list(base_role_info.get("preferred_skills", []))
    min_exp = float(base_role_info.get("typical_exp_years", 2.0))
    keywords = list(base_role_info.get("keywords", []))

    combined_text = f"{query}\n{raw_jd}".strip()

    # Regex extraction from query/JD
    exp_matches = re.findall(r'(\d+(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?)', combined_text, re.IGNORECASE)
    if exp_matches:
        try:
            parsed_exp = float(exp_matches[0])
            if 0.0 <= parsed_exp <= 20.0:
                min_exp = parsed_exp
        except ValueError:
            pass

    text_lower = combined_text.lower()
    for skill_name in ["aws", "docker", "fastapi", "django", "react", "pytorch", "tensorflow", "spark", "kafka", "sql", "nlp", "kubernetes"]:
        if skill_name in text_lower:
            if skill_name.title() not in req_skills and skill_name.title() not in pref_skills:
                pref_skills.append(skill_name.title() if len(skill_name) > 3 else skill_name.upper())

    # Optional LLM JD Parsing
    if raw_jd.strip() and OPENAI_API_KEY:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=OPENAI_API_KEY)
            prompt = f"""
Extract target job requirements from this Job Description into strict JSON:
{{
  "required_skills": ["Skill1", "Skill2"],
  "preferred_skills": ["Skill3"],
  "minimum_experience_years": 3.0,
  "keywords": ["keyword1"]
}}

Job Description:
{raw_jd[:3000]}
"""
            response = client.chat.completions.create(
                model=DEFAULT_LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.1
            )
            import json
            llm_jd = json.loads(response.choices[0].message.content)
            if llm_jd.get("required_skills"):
                req_skills = list(set(req_skills + llm_jd["required_skills"]))
            if llm_jd.get("preferred_skills"):
                pref_skills = list(set(pref_skills + llm_jd["preferred_skills"]))
            if llm_jd.get("minimum_experience_years") is not None:
                min_exp = float(llm_jd["minimum_experience_years"])
        except Exception as e:
            logger.warning(f"LLM JD parsing failed, fallback to heuristic: {e}")

    return JobRequirement(
        role=target_role,
        required_skills=req_skills,
        preferred_skills=pref_skills,
        minimum_experience_years=min_exp,
        keywords=keywords,
        raw_query=query,
        raw_jd=raw_jd
    )


def calculate_role_fit_score(candidate: CandidateProfile, target_role: str, custom_taxonomy: Optional[Dict[str, Any]] = None) -> float:
    taxonomy = custom_taxonomy if custom_taxonomy else JOB_ROLE_TAXONOMY
    role_info = taxonomy.get(target_role, taxonomy.get("Software Engineer", JOB_ROLE_TAXONOMY["Software Engineer"]))
    title_variants = role_info.get("title_variants", [target_role.lower()])
    keywords = role_info.get("keywords", [])

    title_score = 0.4
    cand_titles_lower = [t.lower() for t in candidate.job_titles] + [candidate.headline.lower()]
    for variant in title_variants:
        if any(variant in t for t in cand_titles_lower):
            title_score = 1.0
            break
        elif any(v_word in t for t in cand_titles_lower for v_word in variant.split()):
            title_score = max(title_score, 0.7)

    raw_lower = candidate.raw_text.lower()
    matched_kw = sum(1 for kw in keywords if kw.lower() in raw_lower)
    kw_score = min(1.0, matched_kw / max(1, len(keywords)))

    return (0.6 * title_score) + (0.4 * kw_score)


def calculate_skill_match_score(candidate: CandidateProfile, requirement: JobRequirement) -> Tuple[float, List[str], List[str], List[str], List[str]]:
    cand_skills_lower = set([s.lower() for s in candidate.skills])

    matched_req = []
    missing_req = []
    for req in requirement.required_skills:
        if req.lower() in cand_skills_lower:
            matched_req.append(req)
        else:
            missing_req.append(req)

    req_coverage = len(matched_req) / len(requirement.required_skills) if requirement.required_skills else 1.0

    matched_pref = []
    missing_pref = []
    for pref in requirement.preferred_skills:
        if pref.lower() in cand_skills_lower:
            matched_pref.append(pref)
        else:
            missing_pref.append(pref)

    pref_coverage = len(matched_pref) / len(requirement.preferred_skills) if requirement.preferred_skills else 1.0

    final_skill_score = (0.7 * req_coverage) + (0.3 * pref_coverage)
    return final_skill_score, matched_req, missing_req, matched_pref, missing_pref


def calculate_experience_score(candidate: CandidateProfile, min_exp_required: float) -> Tuple[float, str]:
    cand_exp = candidate.experience_years
    if min_exp_required <= 0.0:
        return 0.9, f"Has {cand_exp:.1f} years experience"

    if cand_exp >= min_exp_required:
        score = 1.0
        reason = f"Meets experience requirement ({cand_exp:.1f} yrs vs {min_exp_required:.1f} yrs required)"
    else:
        ratio = cand_exp / min_exp_required
        score = max(0.2, ratio * 0.8)
        reason = f"Below target experience ({cand_exp:.1f} yrs vs {min_exp_required:.1f} yrs required)"

    return score, reason


def calculate_education_score(candidate: CandidateProfile) -> float:
    if not candidate.education:
        return 0.5

    edu_text = " ".join([f"{e.degree} {e.field_of_study}" for e in candidate.education]).lower()
    if any(term in edu_text for term in ["b.tech", "m.tech", "computer science", "engineering", "b.s", "m.s", "data science"]):
        return 1.0
    return 0.75


def rank_candidates(
    candidates: List[CandidateProfile],
    target_role: str,
    query: str = "",
    raw_jd: str = "",
    search_engine: Optional[SemanticSearchEngine] = None,
    custom_weights: Optional[Dict[str, float]] = None,
    custom_taxonomy: Optional[Dict[str, Any]] = None
) -> List[CandidateMatchResult]:
    if not candidates:
        return []

    weights = custom_weights if custom_weights else DEFAULT_SCORE_WEIGHTS
    requirement = parse_job_requirement(target_role, query, raw_jd, custom_taxonomy)

    if search_engine is None:
        search_engine = SemanticSearchEngine()

    search_query = f"{requirement.role} {query} {raw_jd}".strip()
    semantic_scores = search_engine.compute_similarity(search_query, candidates)

    results: List[CandidateMatchResult] = []

    for candidate in candidates:
        role_fit = calculate_role_fit_score(candidate, target_role, custom_taxonomy)
        skill_score, matched_req, missing_req, matched_pref, missing_pref = calculate_skill_match_score(candidate, requirement)
        exp_score, exp_reason = calculate_experience_score(candidate, requirement.minimum_experience_years)
        sem_score = semantic_scores.get(candidate.id, 0.5)
        edu_score = calculate_education_score(candidate)

        total_score = (
            weights["role_fit"] * role_fit +
            weights["skill_match"] * skill_score +
            weights["experience_match"] * exp_score +
            weights["semantic_relevance"] * sem_score +
            weights["education_match"] * edu_score
        ) * 100.0

        match_reasons = []
        gap_reasons = []

        if requirement.required_skills:
            req_count = len(requirement.required_skills)
            matched_count = len(matched_req)
            match_reasons.append(f"✓ Matched {matched_count}/{req_count} required skills ({', '.join(matched_req)})")
            if missing_req:
                gap_reasons.append(f"○ Missing required skills: {', '.join(missing_req)}")

        if matched_pref:
            match_reasons.append(f"✓ Preferred skills found: {', '.join(matched_pref)}")
        if missing_pref:
            gap_reasons.append(f"○ Preferred skills missing: {', '.join(missing_pref[:4])}")

        if exp_score >= 0.9:
            match_reasons.append(f"✓ {exp_reason}")
        else:
            gap_reasons.append(f"○ {exp_reason}")

        if role_fit > 0.8:
            match_reasons.append(f"✓ Relevant background & title: {candidate.headline}")

        if sem_score >= 0.6:
            match_reasons.append(f"✓ High semantic context relevance ({sem_score*100:.0f}%)")

        breakdown = MatchScoreBreakdown(
            total_score=round(total_score, 1),
            role_fit_score=round(role_fit * 100, 1),
            skill_match_score=round(skill_score * 100, 1),
            experience_score=round(exp_score * 100, 1),
            semantic_score=round(sem_score * 100, 1),
            education_score=round(edu_score * 100, 1),
            matched_required_skills=matched_req,
            missing_required_skills=missing_req,
            matched_preferred_skills=matched_pref,
            missing_preferred_skills=missing_pref,
            match_reasons=match_reasons,
            gap_reasons=gap_reasons
        )

        results.append(CandidateMatchResult(candidate=candidate, score_breakdown=breakdown))

    results.sort(key=lambda r: r.score_breakdown.total_score, reverse=True)
    for idx, res in enumerate(results):
        res.rank = idx + 1

    return results


def anonymize_candidate(candidate: CandidateProfile, rank: int) -> CandidateProfile:
    """Returns a copy of the candidate with PII fields anonymized for Blind Hiring Mode."""
    anon_copy = candidate.model_copy()
    anon_copy.name = f"Candidate #{rank} (Anonymized)"
    anon_copy.email = "anonymized@recruitment.private"
    anon_copy.phone = "CONFIDENTIAL"
    anon_copy.location = "Location Redacted"
    return anon_copy


def generate_interview_questions(candidate: CandidateProfile, target_role: str, breakdown: MatchScoreBreakdown) -> List[str]:
    """Generates tailored interview questions based on candidate skills and identified gaps."""
    questions = []

    # Question on top skill
    top_skills = candidate.skills[:3]
    if top_skills:
        questions.append(f"1. **Architecture & Mastery**: Can you walk us through a complex project where you leveraged **{', '.join(top_skills)}**? What were the primary performance bottlenecks?")

    # Question on missing gap skill
    if breakdown.missing_required_skills or breakdown.missing_preferred_skills:
        missing = (breakdown.missing_required_skills + breakdown.missing_preferred_skills)[0]
        questions.append(f"2. **Gap Probe ({missing})**: Our stack uses **{missing}**. While your resume highlights {', '.join(top_skills[:2])}, how quickly can you onboard onto {missing}?")
    else:
        questions.append(f"2. **System Design & Tradeoffs**: Given your experience in {candidate.headline}, how do you evaluate technology tradeoffs when designing microservices?")

    # Question on experience level
    questions.append(f"3. **Leadership & Impact**: In your {candidate.experience_years:.1f} years of experience, describe a situation where a key production system failed and how you resolved it.")

    return questions
