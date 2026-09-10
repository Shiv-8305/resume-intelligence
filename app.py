"""
Streamlit Web Application for Resume Intelligence & Candidate Matching.
Enterprise Recruiting SaaS with Talent Analytics, Candidate Comparison, Blind Hiring Mode, Export, and Custom Taxonomy Editor.
"""
import os
import io
import pandas as pd
import streamlit as st
from typing import List, Dict

from config import JOB_ROLE_TAXONOMY, OPENAI_API_KEY, DEFAULT_SCORE_WEIGHTS
from models import CandidateProfile
from resume_parser import parse_resume_file
from matcher import rank_candidates, anonymize_candidate, generate_interview_questions
from embeddings import SemanticSearchEngine
from generate_samples import generate_all_samples

# Page Configuration
st.set_page_config(
    page_title="Resume Intelligence ATS",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enterprise Modern SaaS CSS Injection
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .main {
        background-color: #f8fafc;
    }
    
    .header-container {
        padding: 1.2rem 0 0.8rem 0;
        border-bottom: 1px solid #e2e8f0;
        margin-bottom: 1.2rem;
    }
    .header-title {
        font-size: 1.85rem;
        font-weight: 700;
        color: #0f172a;
        margin: 0;
        letter-spacing: -0.02em;
    }
    .header-subtitle {
        font-size: 0.95rem;
        color: #64748b;
        margin-top: 0.25rem;
    }

    .rank-badge {
        background: #f1f5f9;
        color: #334155;
        font-weight: 700;
        font-size: 0.85rem;
        padding: 0.2rem 0.6rem;
        border-radius: 6px;
        border: 1px solid #cbd5e1;
    }
    .candidate-name {
        font-size: 1.25rem;
        font-weight: 700;
        color: #0f172a;
        margin-left: 0.5rem;
    }
    .score-badge {
        background: #ecfdf5;
        color: #047857;
        font-weight: 700;
        font-size: 1.1rem;
        padding: 0.35rem 0.85rem;
        border-radius: 8px;
        border: 1px solid #a7f3d0;
    }
    .headline-text {
        font-size: 0.95rem;
        color: #475569;
        font-weight: 500;
        margin: 0.4rem 0 0.8rem 0;
    }
    .meta-item {
        font-size: 0.85rem;
        color: #64748b;
        margin-right: 1.2rem;
    }
    .skill-pill {
        display: inline-block;
        background: #f8fafc;
        color: #334155;
        font-size: 0.78rem;
        font-weight: 500;
        padding: 0.2rem 0.55rem;
        border-radius: 5px;
        border: 1px solid #e2e8f0;
        margin-right: 0.35rem;
        margin-bottom: 0.35rem;
    }
    .reason-match {
        color: #15803d;
        font-size: 0.85rem;
        font-weight: 500;
        margin-bottom: 0.2rem;
    }
    .reason-gap {
        color: #b91c1c;
        font-size: 0.85rem;
        font-weight: 500;
        margin-bottom: 0.2rem;
    }
    .scanned-warning {
        background: #fffbeb;
        color: #b45309;
        font-size: 0.75rem;
        padding: 0.25rem 0.5rem;
        border-radius: 4px;
        border: 1px solid #fde68a;
    }
</style>
""", unsafe_allow_html=True)


# Initialize Session State
if "candidates" not in st.session_state:
    st.session_state["candidates"] = {}
if "search_engine" not in st.session_state:
    st.session_state["search_engine"] = SemanticSearchEngine()
if "custom_taxonomy" not in st.session_state:
    st.session_state["custom_taxonomy"] = dict(JOB_ROLE_TAXONOMY)
if "custom_weights" not in st.session_state:
    st.session_state["custom_weights"] = dict(DEFAULT_SCORE_WEIGHTS)


# Header Banner
st.markdown("""
<div class="header-container">
    <div class="header-title">Resume Intelligence Enterprise ATS</div>
    <div class="header-subtitle">Generic Resume Parsing & Contextual Candidate Job Matching Engine</div>
</div>
""", unsafe_allow_html=True)


# Sidebar Configuration
with st.sidebar:
    st.markdown("### ⚙️ System Controls")

    # Dynamic API Key state
    user_api_key = st.text_input("OpenAI API Key (Optional)", value=st.session_state.get("openai_api_key", ""), type="password", help="Paste your OpenAI API key to enable LLM parsing & OpenAI embeddings on live web app.")
    if user_api_key:
        st.session_state["openai_api_key"] = user_api_key
        os.environ["OPENAI_API_KEY"] = user_api_key
        import config
        config.OPENAI_API_KEY = user_api_key

    active_key = os.getenv("OPENAI_API_KEY", "") or st.session_state.get("openai_api_key", "")
    if active_key:
        st.success("🟢 OpenAI API: Active (Semantic Embeddings & LLM Enabled)")
    else:
        st.info("🟡 Fallback Mode: Deterministic Regex + TF-IDF Vectorizer Active")

    st.markdown("---")
    st.markdown("### 🛡️ Diversity & Privacy Settings")
    blind_hiring_mode = st.toggle("Blind Hiring Mode (Anonymize PII)", value=False, help="Hides candidate name, contact, and personal identifiers to reduce bias.")

    st.markdown("---")
    st.markdown("### 📊 Index Summary")
    total_parsed = len(st.session_state["candidates"])
    st.metric("Processed Resumes", total_parsed)

    st.markdown("---")
    if st.button("🚀 Load 6 Sample Resumes (1-Click Demo)", use_container_width=True, type="primary"):
        with st.spinner("Generating and indexing sample resumes..."):
            sample_dir = "sample_resumes"
            generate_all_samples(sample_dir)
            loaded_count = 0
            for fname in os.listdir(sample_dir):
                fpath = os.path.join(sample_dir, fname)
                if os.path.isfile(fpath):
                    with open(fpath, "rb") as f:
                        file_bytes = f.read()
                        cand = parse_resume_file(fname, file_bytes)
                        st.session_state["candidates"][cand.id] = cand
                        loaded_count += 1
            st.success(f"Loaded {loaded_count} sample resumes into index!")
            st.rerun()

    if st.button("🗑️ Clear Index", use_container_width=True):
        st.session_state["candidates"] = {}
        st.rerun()


# Four Main Application Tabs
tab_search, tab_compare, tab_analytics, tab_config = st.tabs([
    "🎯 Candidate Search & Matching",
    "⚔️ Side-by-Side Comparison",
    "📈 Talent Analytics & Insights",
    "⚙️ Custom Roles & ATS Weights"
])


# ==============================================================================
# TAB 1: CANDIDATE SEARCH & MATCHING
# ==============================================================================
with tab_search:
    st.subheader("1. Resume Upload & Processing")
    uploaded_files = st.file_uploader(
        "Upload Resumes (PDF, PNG, JPG, WEBP)",
        type=["pdf", "png", "jpg", "jpeg", "webp"],
        accept_multiple_files=True,
        help="Supports PDF documents, scanned PDFs, and image resumes."
    )

    if uploaded_files:
        new_count = 0
        with st.spinner(f"Parsing {len(uploaded_files)} file(s)..."):
            for file in uploaded_files:
                file_bytes = file.read()
                cand = parse_resume_file(file.name, file_bytes)
                st.session_state["candidates"][cand.id] = cand
                new_count += 1
        st.success(f"Successfully processed {new_count} resume(s)!")

    st.markdown("---")
    st.subheader("2. Contextual Search Requirements")

    col_role, col_query = st.columns([1, 2])
    with col_role:
        selected_role = st.selectbox(
            "Target Job Role * (Required)",
            options=list(st.session_state["custom_taxonomy"].keys()),
            index=0
        )
        role_info = st.session_state["custom_taxonomy"][selected_role]
        st.caption(f"**Baseline Skills**: {', '.join(role_info['required_skills'])}")
        st.caption(f"**Expected Exp**: {role_info['typical_exp_years']} years")

    with col_query:
        nl_query = st.text_input(
            "Natural Language Requirements (Optional)",
            placeholder="e.g. Python developers with 3+ years experience, FastAPI and AWS"
        )

    with st.expander("📋 Optional Job Description (Paste full JD text if available)", expanded=False):
        jd_text = st.text_area("Job Description Text", height=100, placeholder="Paste full Job Description here...")

    st.markdown("---")
    st.subheader("3. Ranked Candidate Search Results")

    candidates_list = list(st.session_state["candidates"].values())

    if not candidates_list:
        st.info("No resumes currently in index. Upload PDFs/Images above or click **'🚀 Load 6 Sample Resumes'** in the sidebar to test instantly.")
    else:
        # Run ranking
        match_results = rank_candidates(
            candidates=candidates_list,
            target_role=selected_role,
            query=nl_query,
            raw_jd=jd_text if 'jd_text' in locals() else "",
            search_engine=st.session_state["search_engine"],
            custom_weights=st.session_state["custom_weights"],
            custom_taxonomy=st.session_state["custom_taxonomy"]
        )

        col_count, col_export = st.columns([3, 1])
        with col_count:
            st.markdown(f"**Found {len(match_results)} candidates ranked for '{selected_role}'**")

        with col_export:
            # Export CSV functionality
            export_data = []
            for r in match_results:
                c = r.candidate
                e = r.score_breakdown
                export_data.append({
                    "Rank": r.rank,
                    "Name": c.name,
                    "Total Match Score (%)": e.total_score,
                    "Role Fit Score (%)": e.role_fit_score,
                    "Skill Match Score (%)": e.skill_match_score,
                    "Experience (Years)": c.experience_years,
                    "Skills": ", ".join(c.skills),
                    "Email": c.email
                })
            df_export = pd.DataFrame(export_data)
            csv_bytes = df_export.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Export CSV Report", data=csv_bytes, file_name=f"candidate_ranking_{selected_role.replace(' ', '_')}.csv", mime="text/csv")

        # Display Candidate Cards
        for res in match_results:
            cand = res.candidate
            if blind_hiring_mode:
                cand = anonymize_candidate(cand, res.rank)

            score = res.score_breakdown

            with st.container():
                col1, col2 = st.columns([4, 1])

                with col1:
                    scanned_tag = '<span class="scanned-warning">⚠️ Image/Scanned Resume</span>' if cand.is_scanned else ''
                    st.markdown(f"""
                    <div style="display: flex; align-items: center; margin-bottom: 0.3rem;">
                        <span class="rank-badge">#{res.rank}</span>
                        <span class="candidate-name">{cand.name}</span>
                        <span style="margin-left: 0.8rem;">{scanned_tag}</span>
                    </div>
                    <div class="headline-text">{cand.headline}</div>
                    <div style="margin-bottom: 0.6rem;">
                        <span class="meta-item">⏱️ <strong>{cand.experience_years:.1f} yrs exp</strong></span>
                        <span class="meta-item">📧 {cand.email or 'No email'}</span>
                        <span class="meta-item">📞 {cand.phone or 'No phone'}</span>
                        <span class="meta-item">📄 {cand.filename} ({cand.parsing_method.upper()})</span>
                    </div>
                    """, unsafe_allow_html=True)

                    pill_html = "".join([f'<span class="skill-pill">{s}</span>' for s in cand.skills[:8]])
                    st.markdown(pill_html, unsafe_allow_html=True)

                with col2:
                    st.markdown(f"""
                    <div style="text-align: right; padding-top: 0.5rem;">
                        <div style="font-size: 0.8rem; color: #64748b; font-weight: 500;">MATCH SCORE</div>
                        <div class="score-badge" style="display: inline-block; margin-top: 0.2rem;">{score.total_score:.1f}%</div>
                    </div>
                    """, unsafe_allow_html=True)

                col_reasons, col_gaps = st.columns(2)
                with col_reasons:
                    if score.match_reasons:
                        st.markdown("**Why candidate matches:**")
                        for reason in score.match_reasons:
                            st.markdown(f'<div class="reason-match">{reason}</div>', unsafe_allow_html=True)
                with col_gaps:
                    if score.gap_reasons:
                        st.markdown("**Potential gaps identified:**")
                        for gap in score.gap_reasons:
                            st.markdown(f'<div class="reason-gap">{gap}</div>', unsafe_allow_html=True)

                # Expandable Details
                with st.expander(f"🔍 Detailed Candidate View & ATS Breakdown — {cand.name}", expanded=False):
                    d_col1, d_col2 = st.columns(2)

                    with d_col1:
                        st.markdown("#### ATS Score Breakdown")
                        st.progress(score.role_fit_score / 100.0, text=f"Role Fit: {score.role_fit_score:.1f}%")
                        st.progress(score.skill_match_score / 100.0, text=f"Skill Match: {score.skill_match_score:.1f}%")
                        st.progress(score.experience_score / 100.0, text=f"Experience Match: {score.experience_score:.1f}%")
                        st.progress(score.semantic_score / 100.0, text=f"Semantic Relevance: {score.semantic_score:.1f}%")
                        st.progress(score.education_score / 100.0, text=f"Education Match: {score.education_score:.1f}%")

                    with d_col2:
                        st.markdown("#### Extracted Profile Details")
                        st.write(f"**Candidate Name**: {cand.name}")
                        st.write(f"**Email**: {cand.email}")
                        st.write(f"**Phone**: {cand.phone}")
                        st.write(f"**Total Experience**: {cand.experience_years} years")
                        st.write(f"**All Skills ({len(cand.skills)})**: {', '.join(cand.skills)}")

                    st.markdown("---")
                    st.markdown("#### 🤖 Tailored Interview Question Generator")
                    if st.button(f"Generate Interview Questions for {cand.name}", key=f"q_btn_{cand.id}"):
                        questions = generate_interview_questions(cand, selected_role, score)
                        for q in questions:
                            st.markdown(q)

                    with st.expander("🛠️ View Raw Extracted Structured JSON", expanded=False):
                        st.json(cand.model_dump())

                st.markdown("<hr style='margin: 1.5rem 0; border: 0; border-top: 1px solid #e2e8f0;'>", unsafe_allow_html=True)


# ==============================================================================
# TAB 2: SIDE-BY-SIDE CANDIDATE COMPARISON
# ==============================================================================
with tab_compare:
    st.subheader("⚔️ Candidate Side-by-Side Comparison Matrix")
    if len(st.session_state["candidates"]) < 2:
        st.warning("Please upload or load at least 2 candidates to perform a side-by-side comparison.")
    else:
        cand_options = {c.name: c for c in st.session_state["candidates"].values()}
        cand_names = list(cand_options.keys())

        comp_col1, comp_col2 = st.columns(2)
        with comp_col1:
            name_a = st.selectbox("Select Candidate A", options=cand_names, index=0)
        with comp_col2:
            name_b = st.selectbox("Select Candidate B", options=cand_names, index=min(1, len(cand_names)-1))

        cand_a = cand_options[name_a]
        cand_b = cand_options[name_b]

        st.markdown("---")
        m_col1, m_col2 = st.columns(2)

        with m_col1:
            st.markdown(f"### {cand_a.name}")
            st.write(f"**Headline**: {cand_a.headline}")
            st.write(f"**Experience**: {cand_a.experience_years:.1f} years")
            st.write(f"**Top Skills**: {', '.join(cand_a.skills[:8])}")
            if cand_a.education:
                st.write(f"**Education**: {cand_a.education[0].degree}")

        with m_col2:
            st.markdown(f"### {cand_b.name}")
            st.write(f"**Headline**: {cand_b.headline}")
            st.write(f"**Experience**: {cand_b.experience_years:.1f} years")
            st.write(f"**Top Skills**: {', '.join(cand_b.skills[:8])}")
            if cand_b.education:
                st.write(f"**Education**: {cand_b.education[0].degree}")

        # Skill Overlap Matrix
        st.markdown("### Skill Comparison Matrix")
        all_skills = sorted(list(set(cand_a.skills + cand_b.skills)))
        matrix_data = []
        for s in all_skills:
            matrix_data.append({
                "Skill": s,
                f"{cand_a.name}": "✅ Yes" if s in cand_a.skills else "❌ No",
                f"{cand_b.name}": "✅ Yes" if s in cand_b.skills else "❌ No"
            })
        st.dataframe(pd.DataFrame(matrix_data), use_container_width=True)


# ==============================================================================
# TAB 3: TALENT ANALYTICS & INSIGHTS
# ==============================================================================
with tab_analytics:
    st.subheader("📈 Talent Pool Analytics & Skill Frequency")

    all_cands = list(st.session_state["candidates"].values())
    if not all_cands:
        st.info("No candidates in index. Load sample resumes to view talent analytics.")
    else:
        st_c1, st_c2, st_c3 = st.columns(3)
        st_c1.metric("Total Candidates", len(all_cands))
        avg_exp = sum(c.experience_years for c in all_cands) / max(1, len(all_cands))
        st_c2.metric("Average Experience", f"{avg_exp:.1f} yrs")
        all_skills_flat = [s for c in all_cands for s in c.skills]
        st_c3.metric("Unique Skills Found", len(set(all_skills_flat)))

        st.markdown("---")
        st.markdown("### Top Skills Found Across Candidate Pool")
        skill_counts = pd.Series(all_skills_flat).value_counts().head(12)
        st.bar_chart(skill_counts)


# ==============================================================================
# TAB 4: CUSTOM ROLES & ATS SCORING CONFIGURATOR
# ==============================================================================
with tab_config:
    st.subheader("⚙️ ATS Scoring Weights & Taxonomy Editor")

    st.markdown("### 1. Dynamic Scoring Weight Allocator")
    st.caption("Adjust relative importance of scoring components (Must sum to 100%).")

    w_col1, w_col2 = st.columns(2)
    with w_col1:
        w_role = st.slider("Role Fit Weight (%)", 0, 100, int(st.session_state["custom_weights"]["role_fit"] * 100))
        w_skill = st.slider("Skill Match Weight (%)", 0, 100, int(st.session_state["custom_weights"]["skill_match"] * 100))
        w_exp = st.slider("Experience Weight (%)", 0, 100, int(st.session_state["custom_weights"]["experience_match"] * 100))

    with w_col2:
        w_sem = st.slider("Semantic Relevance Weight (%)", 0, 100, int(st.session_state["custom_weights"]["semantic_relevance"] * 100))
        w_edu = st.slider("Education Weight (%)", 0, 100, int(st.session_state["custom_weights"]["education_match"] * 100))

    total_w = w_role + w_skill + w_exp + w_sem + w_edu
    if total_w != 100:
        st.warning(f"⚠️ Total weights sum to {total_w}%. Please adjust so they sum to exactly 100%.")
    else:
        st.success("✅ Weights correctly sum to 100%.")
        if st.button("Save New Weights"):
            st.session_state["custom_weights"] = {
                "role_fit": w_role / 100.0,
                "skill_match": w_skill / 100.0,
                "experience_match": w_exp / 100.0,
                "semantic_relevance": w_sem / 100.0,
                "education_match": w_edu / 100.0
            }
            st.success("Updated ATS scoring weights!")

    st.markdown("---")
    st.markdown("### 2. Add Custom Job Role to Taxonomy")

    with st.form("new_role_form"):
        new_role_name = st.text_input("Role Title", placeholder="e.g. Cybersecurity Engineer")
        new_req_skills = st.text_input("Required Skills (comma separated)", placeholder="Python, Linux, Cryptography")
        new_pref_skills = st.text_input("Preferred Skills (comma separated)", placeholder="AWS, Wireshark, SIEM")
        new_exp_years = st.number_input("Typical Required Experience (Years)", min_value=0.0, max_value=20.0, value=3.0)

        submit_role = st.form_submit_button("Add Custom Job Role")
        if submit_role and new_role_name.strip():
            req_list = [s.strip() for s in new_req_skills.split(",") if s.strip()]
            pref_list = [s.strip() for s in new_pref_skills.split(",") if s.strip()]
            st.session_state["custom_taxonomy"][new_role_name] = {
                "required_skills": req_list,
                "preferred_skills": pref_list,
                "keywords": req_list + pref_list,
                "title_variants": [new_role_name.lower()],
                "typical_exp_years": new_exp_years,
                "description": f"Custom recruiter defined role for {new_role_name}."
            }
            st.success(f"Added '{new_role_name}' to Job Role Taxonomy!")
            st.rerun()
