"""
Generates synthetic sample PDF resumes for instant demonstration and testing.
"""
import os
import pymupdf as fitz


def create_pdf_resume(filepath: str, text_content: str):
    """Creates a clean formatted PDF file from plain text using PyMuPDF."""
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)  # A4 size
    rect = fitz.Rect(50, 50, 545, 792)

    # Insert text into page
    page.insert_textbox(rect, text_content, fontsize=10, fontname="helv")
    doc.save(filepath)
    doc.close()


def generate_all_samples(output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    # Sample 1: Rahul Sharma - Senior Python & FastAPI Developer
    rahul_text = """RAHUL SHARMA
Email: rahul.sharma@example.com | Phone: +91 98765 43210 | Location: Bengaluru, India
Professional Headline: Senior Python & Backend Developer

SUMMARY
Resourceful Backend Developer with 4.2 years of experience designing scalable microservices, RESTful APIs, and database solutions using Python, FastAPI, Django, and PostgreSQL. Experienced in AWS cloud deployment and Docker containerization.

EXPERIENCE
Senior Python Developer | TechCraft Solutions (2022 - Present)
- Architected asynchronous REST APIs using FastAPI handling 5M+ daily requests.
- Optimized PostgreSQL queries reducing latency by 40%.
- Integrated Redis caching and deployed microservices on AWS ECS using Docker.

Python Developer | CloudData Systems (2020 - 2022)
- Built automated ETL data processing pipelines in Python and Pandas.
- Implemented JWT authentication and CI/CD pipelines with GitHub Actions.

SKILLS
Python, FastAPI, Django, REST API, SQL, PostgreSQL, Docker, AWS, Redis, Git, Microservices, PyTest

EDUCATION
B.Tech in Computer Science & Engineering | VTU Bengaluru (2016 - 2020)
"""
    create_pdf_resume(os.path.join(output_dir, "rahul_sharma_python_dev.pdf"), rahul_text)

    # Sample 2: Ananya Rao - AI Intern
    ananya_text = """ANANYA RAO
Email: ananya.rao@example.com | Phone: +91 91234 56789 | Location: Hyderabad, India
Professional Headline: AI Intern & NLP Enthusiast

SUMMARY
Passionate Computer Science graduate student seeking an AI Intern role. 0.8 years of hands-on project experience in Natural Language Processing, Generative AI, PyTorch, and Transformer models.

EXPERIENCE
AI Research Intern | DataMind Labs (2024 - Present)
- Assisted in fine-tuning Llama-3 models on custom domain datasets using PyTorch.
- Built interactive RAG prototypes using LangChain and Python.

SKILLS
Python, Machine Learning, Deep Learning, NLP, Generative AI, LLM, PyTorch, Pandas, Git

PROJECTS
- Sentiment Analysis Engine: Trained BERT classifier achieving 92% accuracy.
- Document Q&A Bot: Implemented vector embeddings and cosine similarity search.

EDUCATION
M.S. in Artificial Intelligence | IIIT Hyderabad (2023 - 2025)
B.S. in Computer Science | Osmania University (2019 - 2023)
"""
    create_pdf_resume(os.path.join(output_dir, "ananya_rao_ai_intern.pdf"), ananya_text)

    # Sample 3: Vikram Patel - Data Scientist
    vikram_text = """VIKRAM PATEL
Email: vikram.patel@example.com | Phone: +91 99887 76655 | Location: Mumbai, India
Professional Headline: Senior Data Scientist

SUMMARY
Data Scientist with 3.5 years of experience applying machine learning, statistical modeling, and predictive analytics to enterprise business problems. Expert in Python, SQL, Scikit-Learn, and Tableau.

EXPERIENCE
Data Scientist | FinAnalytics Corp (2021 - Present)
- Developed customer churn prediction model with XGBoost saving $1.2M annually.
- Conducted A/B testing and engineered feature extraction pipelines in Python and SQL.

SKILLS
Python, SQL, Statistics, Machine Learning, Pandas, NumPy, Scikit-Learn, Tableau, PowerBI, R

EDUCATION
B.Tech in Statistics & Data Science | IIT Bombay (2017 - 2021)
"""
    create_pdf_resume(os.path.join(output_dir, "vikram_patel_data_scientist.pdf"), vikram_text)

    # Sample 4: Priya Sundaram - Java Backend Engineer
    priya_text = """PRIYA SUNDARAM
Email: priya.sundaram@example.com | Phone: +91 97654 32109 | Location: Chennai, India
Professional Headline: Java Backend Engineer

SUMMARY
Experienced Java Engineer with 5.0 years in building high-throughput enterprise systems, microservices, and financial transaction platforms using Java 17, Spring Boot, Hibernate, Kafka, and PostgreSQL.

EXPERIENCE
Lead Java Engineer | Global Pay Systems (2021 - Present)
- Led team of 6 engineers developing real-time payment gateway using Spring Boot and Kafka.
- Containerized applications with Docker and Kubernetes on AWS.

Java Developer | Infosolutions (2019 - 2021)
- Developed enterprise REST APIs with Java, Maven, and Spring Security.

SKILLS
Java, Spring Boot, Microservices, Hibernate, SQL, PostgreSQL, Maven, Docker, REST API, Kafka, Git

EDUCATION
B.Tech in Information Technology | Anna University (2015 - 2019)
"""
    create_pdf_resume(os.path.join(output_dir, "priya_sundaram_java_dev.pdf"), priya_text)

    # Sample 5: Alex Chen - GenAI Engineer
    alex_text = """ALEX CHEN
Email: alex.chen@example.com | Phone: +1 415 555 0199 | Location: San Francisco, CA
Professional Headline: Generative AI & RAG Specialist

SUMMARY
GenAI Engineer with 3.0 years of software experience specializing in Retrieval Augmented Generation (RAG), vector databases (Pinecone, Chroma), LangChain, and OpenAI API integrations.

EXPERIENCE
GenAI Engineer | AI Scale Labs (2022 - Present)
- Built enterprise Knowledge Assistant using LlamaIndex, Python, FastAPI, and Qdrant.
- Optimized prompt templates and LLM evaluation benchmarks for enterprise clients.

SKILLS
Python, Generative AI, LLM, LangChain, LlamaIndex, Vector DB, RAG, Prompt Engineering, FastAPI, PyTorch, Git

EDUCATION
B.S. in Computer Science | UC Berkeley (2018 - 2022)
"""
    create_pdf_resume(os.path.join(output_dir, "alex_chen_genai_engineer.pdf"), alex_text)

    # Sample 6: Scanned Resume Test
    scanned_text = """SCAN"""
    create_pdf_resume(os.path.join(output_dir, "scanned_resume_sample.pdf"), scanned_text)

    print(f"Successfully generated 6 sample resumes in '{output_dir}'.")


if __name__ == "__main__":
    generate_all_samples("sample_resumes")
