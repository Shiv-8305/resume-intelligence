"""
Configuration & Job Role Taxonomy for Resume Intelligence ATS.
"""
import os
from typing import Dict, List, Any

# Environment & Model Settings
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_LLM_MODEL = "gpt-4o-mini"

# Hybrid Scoring Weights (Must sum to 1.0)
DEFAULT_SCORE_WEIGHTS = {
    "role_fit": 0.35,
    "skill_match": 0.30,
    "experience_match": 0.20,
    "semantic_relevance": 0.10,
    "education_match": 0.05,
}

# Configurable Job Role Taxonomy
JOB_ROLE_TAXONOMY: Dict[str, Dict[str, Any]] = {
    "Python Developer": {
        "required_skills": ["Python", "REST API", "Git"],
        "preferred_skills": ["FastAPI", "Django", "Flask", "SQL", "PostgreSQL", "Docker", "AWS", "Redis"],
        "keywords": ["backend", "microservices", "web development", "asyncio", "pytest"],
        "title_variants": ["python developer", "backend developer", "python engineer", "software developer", "software engineer"],
        "typical_exp_years": 3.0,
        "description": "Develops scalable backend services, APIs, and data processing pipelines using Python."
    },
    "Java Developer": {
        "required_skills": ["Java", "Spring Boot", "Git"],
        "preferred_skills": ["Microservices", "Hibernate", "SQL", "Maven", "Docker", "REST API", "Kafka", "PostgreSQL"],
        "keywords": ["backend", "enterprise", "object oriented", "jvm", "jpa"],
        "title_variants": ["java developer", "java engineer", "backend developer", "software developer", "software engineer"],
        "typical_exp_years": 3.0,
        "description": "Builds robust enterprise applications and microservices using Java and Spring ecosystem."
    },
    "Software Engineer": {
        "required_skills": ["Software Development", "Data Structures", "Git"],
        "preferred_skills": ["Python", "Java", "C++", "SQL", "REST API", "Docker", "CI/CD", "System Design"],
        "keywords": ["problem solving", "algorithms", "backend", "full stack", "architecture"],
        "title_variants": ["software engineer", "software developer", "sde", "member of technical staff", "systems engineer"],
        "typical_exp_years": 2.0,
        "description": "Generalist software engineering role focused on core system design and application development."
    },
    "AI/ML Engineer": {
        "required_skills": ["Python", "Machine Learning", "PyTorch"],
        "preferred_skills": ["TensorFlow", "Scikit-Learn", "NLP", "Computer Vision", "Docker", "MLOps", "SQL", "Pandas"],
        "keywords": ["model training", "neural networks", "deep learning", "feature engineering", "model deployment"],
        "title_variants": ["ai/ml engineer", "ml engineer", "machine learning engineer", "ai engineer", "data scientist"],
        "typical_exp_years": 3.0,
        "description": "Designs, trains, and deploys machine learning models and AI pipelines."
    },
    "AI Intern": {
        "required_skills": ["Python"],
        "preferred_skills": ["Machine Learning", "NLP", "Deep Learning", "Generative AI", "LLM", "PyTorch", "TensorFlow", "Pandas"],
        "keywords": ["internship", "research", "fine-tuning", "prompt engineering", "experiments", "computer science"],
        "title_variants": ["ai intern", "ml intern", "machine learning intern", "research intern", "ai research intern", "data science intern"],
        "typical_exp_years": 0.5,
        "description": "Entry-level or student role focusing on AI research, experimentation, and ML prototype building."
    },
    "Data Scientist": {
        "required_skills": ["Python", "SQL", "Statistics"],
        "preferred_skills": ["Machine Learning", "Pandas", "NumPy", "Scikit-Learn", "Tableau", "PowerBI", "A/B Testing", "R"],
        "keywords": ["predictive modeling", "data analysis", "hypothesis testing", "visualization", "insights"],
        "title_variants": ["data scientist", "senior data scientist", "applied scientist", "quantitative analyst"],
        "typical_exp_years": 3.0,
        "description": "Analyzes complex datasets to extract business insights and build predictive algorithms."
    },
    "Data Engineer": {
        "required_skills": ["Python", "SQL", "Data Pipelines"],
        "preferred_skills": ["Spark", "Airflow", "ETL", "Kafka", "Snowflake", "BigQuery", "AWS", "Docker"],
        "keywords": ["data warehousing", "data architecture", "orchestration", "streaming", "batch processing"],
        "title_variants": ["data engineer", "etl developer", "big data engineer", "data platform engineer"],
        "typical_exp_years": 3.0,
        "description": "Constructs and maintains robust data architecture, data lakes, and ETL pipelines."
    },
    "Data Analyst": {
        "required_skills": ["SQL", "Excel"],
        "preferred_skills": ["Python", "Tableau", "PowerBI", "R", "Statistics", "Data Visualization", "Pandas"],
        "keywords": ["reporting", "dashboarding", "business intelligence", "metrics", "analytics"],
        "title_variants": ["data analyst", "bi analyst", "business intelligence analyst", "reporting analyst"],
        "typical_exp_years": 2.0,
        "description": "Transforms raw data into actionable dashboards and operational reports."
    },
    "GenAI Engineer": {
        "required_skills": ["Python", "Generative AI"],
        "preferred_skills": ["LLM", "LangChain", "LlamaIndex", "Vector DB", "RAG", "Prompt Engineering", "FastAPI", "PyTorch"],
        "keywords": ["retrieval augmented generation", "embeddings", "fine-tuning", "agents", "openai", "transformers"],
        "title_variants": ["genai engineer", "generative ai engineer", "llm engineer", "ai engineer"],
        "typical_exp_years": 2.0,
        "description": "Specializes in building LLM-powered applications, RAG pipelines, and agentic workflows."
    },
    "Backend Developer": {
        "required_skills": ["REST API", "SQL", "Git"],
        "preferred_skills": ["Python", "Node.js", "Java", "Go", "PostgreSQL", "Docker", "Redis", "Microservices", "AWS"],
        "keywords": ["server-side", "database design", "authentication", "scalability", "api gateway"],
        "title_variants": ["backend developer", "backend engineer", "server engineer", "software engineer"],
        "typical_exp_years": 3.0,
        "description": "Focuses on server-side business logic, APIs, database integration, and security."
    },
    "Full Stack Developer": {
        "required_skills": ["JavaScript", "HTML/CSS", "Git"],
        "preferred_skills": ["React", "Node.js", "TypeScript", "Python", "SQL", "PostgreSQL", "REST API", "Docker", "Tailwind"],
        "keywords": ["frontend", "backend", "web app", "ui/ux", "database", "full stack"],
        "title_variants": ["full stack developer", "full stack engineer", "web developer", "software engineer"],
        "typical_exp_years": 3.0,
        "description": "Handles both client-side UI frontend development and server-side backend logic."
    }
}
