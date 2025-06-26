import os
import fitz  # PyMuPDF
from dotenv import load_dotenv
from typing import Dict, List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()


# Configuration
TEXT_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", " ", ""]
)
EMBEDDINGS = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001"
)

RESUME_TEMPLATES = {
    "template_resume": """Analyze this resume and extract key professional information:
{text}

Output format:
- Core Professional Summary (3-4 sentences)
- Key Achievements (bulleted list)
- Education Background""",
    
    "template_skills_resume": """Extract all technical skills and certifications from this resume:
{text}

Format as:
### TECHNICAL SKILLS
- [Category]: [Skill1], [Skill2]

### CERTIFICATIONS
- [Name] ([Issuer], [Year])""",
    
    "template_exp_resume": """Extract work experience details from this resume:
{text}

For each position include:
- Job Title @ Company (Dates)
- Key Responsibilities (3 bullet points)
- Technologies Used"""
}

JD_TEMPLATES = {
    "template_jd": """Analyze this job description and extract key requirements:
{text}

Output format:
- Role Summary (3-4 sentences)
- Key Responsibilities
- Performance Expectations""",
    
    "template_skills_jd": """Extract all required technical skills and preferred certifications:
{text}

Format as:
### REQUIRED SKILLS
- [Category]: [Skill1], [Skill2]

### PREFERRED CERTIFICATIONS
- [Name] ([Issuer if specified])""",
    
    "template_exp_jd": """Extract experience requirements from this job description:
{text}

Include:
- Minimum Years Experience
- Specific Role Requirements
- Industry Preferences"""
}

def pdf_to_markdown(file_path: str) -> str:
    """Convert PDF to plain text"""
    doc = fitz.open(file_path)
    plain_text = ""
    for page in doc:
        text_dict = page.get_text("dict")
        for block in text_dict["blocks"]:
            if "lines" in block:
                for line in block["lines"]:
                    for span in line["spans"]:
                        plain_text += span["text"]
                    plain_text += "\n"
                plain_text += "\n"
    return plain_text

def process_text(text: str) -> List[str]:
    """Split text into chunks"""
    return TEXT_SPLITTER.split_text(text)

def create_faiss_index(chunks: List[str]) -> FAISS:
    """Create FAISS vector store from text chunks"""
    return FAISS.from_texts(chunks, EMBEDDINGS)

def calculate_similarity(resume_index: FAISS, jd_index: FAISS) -> float:
    """Calculate cosine similarity between resume and JD"""
    jd_embeddings = jd_index.index.reconstruct_n(0, jd_index.index.ntotal)
    avg_jd_embedding = np.mean(jd_embeddings, axis=0).reshape(1, -1)
    
    scores = []
    for i in range(resume_index.index.ntotal):
        resume_embedding = resume_index.index.reconstruct(i).reshape(1, -1)
        score = cosine_similarity(avg_jd_embedding, resume_embedding)[0][0]
        scores.append(score)
    
    return float(np.mean(scores))

def apply_template(text: str, template: str, is_resume: bool = True) -> str:
    """Apply prompt template to text using LLM with error handling and debug prints"""
    templates = RESUME_TEMPLATES if is_resume else JD_TEMPLATES
    prompt = PromptTemplate.from_template(templates[template])
    
    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)
    
    chain = prompt | model
    try:
        response = chain.invoke({"text": text})
        content = response.content if hasattr(response, "content") else str(response)
        print(f"\n[Template: {template}] Output:\n{content[:500]}")  # Print first 500 chars
        if not content.strip():
            print(f"Warning: Empty output for template {template}")
        return content
    except Exception as e:
        print(f"Error processing template {template}: {str(e)}")
        return ""

def enhanced_screen_resume(resume_path: str, job_description: str) -> Dict:
    """Enhanced resume screening with template-based analysis and debug prints"""
    try:
        resume_md = pdf_to_markdown(resume_path)
        print(f"\nExtracted Resume Text (first 500 chars):\n{resume_md[:500]}")
        print(f"\nJob Description Text:\n{job_description[:500]}")

        resume_outputs = {
            name: apply_template(resume_md, name, is_resume=True)
            for name in RESUME_TEMPLATES
        }
        
        jd_outputs = {
            name: apply_template(job_description, name, is_resume=False)
            for name in JD_TEMPLATES
        }
        
        print("\nResume Template Outputs Summary:")
        for k, v in resume_outputs.items():
            print(f" - {k}: {len(v)} chars")
        print("\nJob Description Template Outputs Summary:")
        for k, v in jd_outputs.items():
            print(f" - {k}: {len(v)} chars")
        
        embeddings = {
            "resume": {name: create_faiss_index(process_text(text)) 
                       for name, text in resume_outputs.items()},
            "jd": {name: create_faiss_index(process_text(text)) 
                   for name, text in jd_outputs.items()}
        }
        print("\nEmbeddings created for resume and JD templates.")
        
        similarity_results = {
            "overall": calculate_similarity(embeddings["resume"]["template_resume"], 
                                          embeddings["jd"]["template_jd"]),
            "skills": calculate_similarity(embeddings["resume"]["template_skills_resume"], 
                                         embeddings["jd"]["template_skills_jd"]),
            "experience": calculate_similarity(embeddings["resume"]["template_exp_resume"], 
                                            embeddings["jd"]["template_exp_jd"])
        }
        print(f"\nSimilarity Results:\n{similarity_results}")
        
        return {
            "scores": similarity_results,
            "resume_analysis": resume_outputs,
            "jd_analysis": jd_outputs
        }
        
    except Exception as e:
        print(f"Error in enhanced screening: {str(e)}")
        return {}
