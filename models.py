"""
Data models and schemas for candidate profiles, JD requirements, and ATS match scoring.
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class WorkExperience(BaseModel):
    title: str = ""
    company: str = ""
    duration: str = ""
    description: str = ""


class EducationItem(BaseModel):
    degree: str = ""
    institution: str = ""
    year: str = ""
    field_of_study: str = ""


class CandidateProfile(BaseModel):
    id: str
    filename: str
    file_type: str = "pdf"  # "pdf", "image"
    name: str = "Unknown Candidate"
    email: str = ""
    phone: str = ""
    location: str = ""
    headline: str = ""
    job_titles: List[str] = Field(default_factory=list)
    experience_years: float = 0.0
    skills: List[str] = Field(default_factory=list)
    education: List[EducationItem] = Field(default_factory=list)
    certifications: List[str] = Field(default_factory=list)
    projects: List[str] = Field(default_factory=list)
    work_experience: List[WorkExperience] = Field(default_factory=list)
    achievements: List[str] = Field(default_factory=list)
    languages: List[str] = Field(default_factory=list)
    raw_text: str = ""
    is_scanned: bool = False
    parsing_method: str = "deterministic"  # "llm" or "deterministic"


class JobRequirement(BaseModel):
    role: str
    required_skills: List[str] = Field(default_factory=list)
    preferred_skills: List[str] = Field(default_factory=list)
    minimum_experience_years: float = 0.0
    education_fields: List[str] = Field(default_factory=list)
    keywords: List[str] = Field(default_factory=list)
    seniority: str = "any"
    raw_query: str = ""
    raw_jd: str = ""


class MatchScoreBreakdown(BaseModel):
    total_score: float  # Percentage 0 - 100
    role_fit_score: float
    skill_match_score: float
    experience_score: float
    semantic_score: float
    education_score: float
    matched_required_skills: List[str] = Field(default_factory=list)
    missing_required_skills: List[str] = Field(default_factory=list)
    matched_preferred_skills: List[str] = Field(default_factory=list)
    missing_preferred_skills: List[str] = Field(default_factory=list)
    match_reasons: List[str] = Field(default_factory=list)
    gap_reasons: List[str] = Field(default_factory=list)


class CandidateMatchResult(BaseModel):
    candidate: CandidateProfile
    score_breakdown: MatchScoreBreakdown
    rank: int = 0
