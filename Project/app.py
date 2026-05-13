from flask import Flask, request, render_template, jsonify
import os
import fitz
import requests
import json
import re
import time
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT
import io
from datetime import datetime
from typing import Dict, List, Any
import uuid
from dataclasses import dataclass, asdict
from enum import Enum
from queue import Queue
import threading

load_dotenv()

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
ANALYSIS_RESULTS_FOLDER = "analysis_results"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(ANALYSIS_RESULTS_FOLDER, exist_ok=True)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = "gemini-2.5-flash"

if not GEMINI_API_KEY:
    print("⚠️  WARNING: GEMINI_API_KEY not found in .env file!")
    GEMINI_API_KEY = "NOT_SET"

# ==================== AGENTIC FRAMEWORK ====================

class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"

class TaskType(Enum):
    PDF_EXTRACTION = "pdf_extraction"
    QUANTUM_IR_CHECK = "quantum_ir_check"
    EQUATION_EXTRACTION = "equation_extraction"
    SYMBOL_ANALYSIS = "symbol_analysis"
    CONCEPT_IDENTIFICATION = "concept_identification"
    TERM_HIGHLIGHTING = "term_highlighting"
    PDF_GENERATION = "pdf_generation"
    METADATA_ENRICHMENT = "metadata_enrichment"

@dataclass
class Task:
    task_id: str
    task_type: TaskType
    status: TaskStatus
    input_data: Dict[str, Any]
    output_data: Dict[str, Any] = None
    error_message: str = None
    created_at: str = None
    completed_at: str = None
    retry_count: int = 0
    max_retries: int = 3

@dataclass
class AnalysisJob:
    job_id: str
    filename: str
    status: TaskStatus
    created_at: str
    tasks: List[Task] = None
    final_results: Dict[str, Any] = None
    error_log: List[str] = None
    
    def __post_init__(self):
        if self.tasks is None:
            self.tasks = []
        if self.error_log is None:
            self.error_log = []

# Global state management
analysis_jobs: Dict[str, AnalysisJob] = {}
task_queue = Queue()
results_cache: Dict[str, Any] = {}

# Quantum IR Glossary
QUANTUM_IR_GLOSSARY = {
    "qubit": "A quantum bit - the basic unit of quantum information that can be 0, 1, or both simultaneously",
    "superposition": "The ability of a quantum system to be in multiple states at once until measured",
    "entanglement": "A quantum phenomenon where particles become connected and affect each other instantly",
    "quantum state": "The mathematical description of a quantum system, often written as |ψ⟩",
    "probability amplitude": "A complex number that determines the probability of measuring a particular outcome",
    "hilbert space": "The mathematical space where quantum states live",
    "unitary operator": "A mathematical operation that preserves quantum information",
    "measurement": "The act of observing a quantum system, which causes it to collapse to a definite state",
    "grover's algorithm": "A quantum search algorithm that finds items in unsorted databases faster than classical methods",
    "quantum ranking": "Using quantum principles to rank search results or documents",
    "density matrix": "A mathematical tool to describe quantum states, especially mixed states",
    "tensor product": "A way to combine quantum systems mathematically",
    "bra-ket notation": "The notation |⟩ and ⟨| used in quantum mechanics to represent states",
    "hermitian": "A type of matrix that has real eigenvalues, important in quantum mechanics",
    "eigenvalue": "A special value that results from applying an operator to a vector",
    "eigenvector": "A vector that only gets scaled (not rotated) when an operator is applied"
}

# ==================== AGENT CLASS ====================

class QuantumIRAgent:
    """Autonomous agent that manages analysis workflow"""
    
    def __init__(self):
        self.name = "Quantum IR Analyzer Agent"
        self.version = "2.0"
    
    def create_analysis_workflow(self, filename: str) -> str:
        """Create a complete analysis workflow"""
        job_id = str(uuid.uuid4())
        job = AnalysisJob(
            job_id=job_id,
            filename=filename,
            status=TaskStatus.PENDING,
            created_at=datetime.now().isoformat()
        )
        analysis_jobs[job_id] = job
        
        # Create task sequence
        tasks = [
            Task(
                task_id=f"{job_id}_1",
                task_type=TaskType.PDF_EXTRACTION,
                status=TaskStatus.PENDING,
                input_data={"filename": filename},
                created_at=datetime.now().isoformat()
            ),
            Task(
                task_id=f"{job_id}_2",
                task_type=TaskType.QUANTUM_IR_CHECK,
                status=TaskStatus.PENDING,
                input_data={"job_id": job_id},
                created_at=datetime.now().isoformat()
            ),
            Task(
                task_id=f"{job_id}_3",
                task_type=TaskType.EQUATION_EXTRACTION,
                status=TaskStatus.PENDING,
                input_data={"job_id": job_id},
                created_at=datetime.now().isoformat()
            ),
            Task(
                task_id=f"{job_id}_4",
                task_type=TaskType.SYMBOL_ANALYSIS,
                status=TaskStatus.PENDING,
                input_data={"job_id": job_id},
                created_at=datetime.now().isoformat()
            ),
            Task(
                task_id=f"{job_id}_5",
                task_type=TaskType.CONCEPT_IDENTIFICATION,
                status=TaskStatus.PENDING,
                input_data={"job_id": job_id},
                created_at=datetime.now().isoformat()
            ),
            Task(
                task_id=f"{job_id}_6",
                task_type=TaskType.TERM_HIGHLIGHTING,
                status=TaskStatus.PENDING,
                input_data={"job_id": job_id},
                created_at=datetime.now().isoformat()
            ),
            Task(
                task_id=f"{job_id}_7",
                task_type=TaskType.METADATA_ENRICHMENT,
                status=TaskStatus.PENDING,
                input_data={"job_id": job_id},
                created_at=datetime.now().isoformat()
            ),
            Task(
                task_id=f"{job_id}_8",
                task_type=TaskType.PDF_GENERATION,
                status=TaskStatus.PENDING,
                input_data={"job_id": job_id},
                created_at=datetime.now().isoformat()
            )
        ]
        
        job.tasks = tasks
        analysis_jobs[job_id] = job
        
        return job_id
    
    def execute_workflow(self, job_id: str) -> bool:
        """Execute analysis workflow sequentially"""
        job = analysis_jobs.get(job_id)
        if not job:
            return False
        
        job.status = TaskStatus.IN_PROGRESS
        
        for task in job.tasks:
            try:
                self.execute_task(task, job)
                if task.status == TaskStatus.FAILED:
                    job.status = TaskStatus.FAILED
                    return False
            except Exception as e:
                task.status = TaskStatus.FAILED
                task.error_message = str(e)
                job.error_log.append(f"Task {task.task_id} failed: {str(e)}")
                job.status = TaskStatus.FAILED
                return False
        
        job.status = TaskStatus.COMPLETED
        job.final_results = self.compile_results(job)
        return True
    
    def execute_task(self, task: Task, job: AnalysisJob):
        """Execute individual task"""
        task.status = TaskStatus.IN_PROGRESS
        
        try:
            if task.task_type == TaskType.PDF_EXTRACTION:
                self._handle_pdf_extraction(task, job)
            elif task.task_type == TaskType.QUANTUM_IR_CHECK:
                self._handle_quantum_ir_check(task, job)
            elif task.task_type == TaskType.EQUATION_EXTRACTION:
                self._handle_equation_extraction(task, job)
            elif task.task_type == TaskType.SYMBOL_ANALYSIS:
                self._handle_symbol_analysis(task, job)
            elif task.task_type == TaskType.CONCEPT_IDENTIFICATION:
                self._handle_concept_identification(task, job)
            elif task.task_type == TaskType.TERM_HIGHLIGHTING:
                self._handle_term_highlighting(task, job)
            elif task.task_type == TaskType.METADATA_ENRICHMENT:
                self._handle_metadata_enrichment(task, job)
            elif task.task_type == TaskType.PDF_GENERATION:
                self._handle_pdf_generation(task, job)
            
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now().isoformat()
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error_message = str(e)
            task.retry_count += 1
            if task.retry_count < task.max_retries:
                task.status = TaskStatus.PENDING
    
    # Task Handlers
    def _handle_pdf_extraction(self, task: Task, job: AnalysisJob):
        """Extract text from PDF"""
        filename = task.input_data.get("filename")
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        
        pdf_text = extract_text_from_pdf(filepath)
        
        task.output_data = {
            "pdf_text": pdf_text,
            "text_length": len(pdf_text),
            "filepath": filepath
        }
        
        results_cache[f"{job.job_id}_pdf_text"] = pdf_text
        results_cache[f"{job.job_id}_filepath"] = filepath
    
    def _handle_quantum_ir_check(self, task: Task, job: AnalysisJob):
        """Check if PDF is related to Quantum IR"""
        pdf_text = results_cache.get(f"{job.job_id}_pdf_text", "")
        is_quantum_ir, keywords = check_quantum_ir(pdf_text)
        
        task.output_data = {
            "is_quantum_ir": is_quantum_ir,
            "keywords": keywords,
            "confidence": len(keywords) / 9 * 100
        }
        
        results_cache[f"{job.job_id}_is_quantum_ir"] = is_quantum_ir
        results_cache[f"{job.job_id}_keywords"] = keywords
    
    def _handle_equation_extraction(self, task: Task, job: AnalysisJob):
        """Extract equations using Gemini"""
        pdf_text = results_cache.get(f"{job.job_id}_pdf_text", "")
        filepath = results_cache.get(f"{job.job_id}_filepath", "")
        filename = os.path.basename(filepath)
        
        equations = extract_equations_with_gemini(pdf_text, filename)
        
        task.output_data = {
            "equations_count": len(equations),
            "equations": equations
        }
        
        results_cache[f"{job.job_id}_equations"] = equations
    
    def _handle_symbol_analysis(self, task: Task, job: AnalysisJob):
        """Analyze symbols in equations"""
        equations = results_cache.get(f"{job.job_id}_equations", [])
        
        for eq in equations:
            if "symbols" not in eq:
                symbols = extract_symbols_from_equation(eq.get("equation", ""))
                eq["symbols"] = symbols
                eq["symbol_definitions"] = generate_symbol_definitions(
                    symbols, 
                    eq.get("equation", ""), 
                    eq.get("explanation", "")
                )
        
        task.output_data = {
            "symbols_analyzed": True,
            "equations_count": len(equations)
        }
    
    def _handle_concept_identification(self, task: Task, job: AnalysisJob):
        """Identify quantum concepts"""
        equations = results_cache.get(f"{job.job_id}_equations", [])
        
        all_concepts = []
        for eq in equations:
            concepts = identify_quantum_concepts(
                eq.get("equation", ""), 
                eq.get("explanation", "")
            )
            eq["concepts"] = concepts
            all_concepts.extend(concepts)
        
        task.output_data = {
            "concepts_identified": True,
            "unique_concepts": list(set(all_concepts)),
            "concept_count": len(set(all_concepts))
        }
    
    def _handle_term_highlighting(self, task: Task, job: AnalysisJob):
        """Highlight quantum terms"""
        equations = results_cache.get(f"{job.job_id}_equations", [])
        pdf_text = results_cache.get(f"{job.job_id}_pdf_text", "")
        
        for eq in equations:
            # Highlight terms in explanation
            explanation = eq.get("explanation", "")
            highlighted = []
            
            # Find glossary terms in this equation's explanation
            for term, definition in QUANTUM_IR_GLOSSARY.items():
                if term.lower() in explanation.lower():
                    highlighted.append({
                        "term": term,
                        "definition": definition
                    })
            
            # Also check in the equation text itself
            equation_text = eq.get("equation", "").replace('$', '').replace('\\', '')
            for term, definition in QUANTUM_IR_GLOSSARY.items():
                if term.lower() in equation_text.lower():
                    # Check if term already added
                    if not any(h["term"] == term for h in highlighted):
                        highlighted.append({
                            "term": term,
                            "definition": definition
                        })
            
            eq["glossary_terms"] = highlighted
        
        task.output_data = {
            "terms_highlighted": True,
            "document_terms_count": len(highlight_quantum_terms(pdf_text)),
            "equations_processed": len(equations)
    }
    
    def _handle_metadata_enrichment(self, task: Task, job: AnalysisJob):
        """Enrich results with metadata"""
        equations = results_cache.get(f"{job.job_id}_equations", [])
        is_quantum_ir = results_cache.get(f"{job.job_id}_is_quantum_ir", False)
        
        enriched_data = {
            "total_equations": len(equations),
            "analysis_timestamp": datetime.now().isoformat(),
            "job_id": job.job_id,
            "is_quantum_ir": is_quantum_ir,
            "gemini_model": GEMINI_MODEL,
            "glossary_terms_used": len(QUANTUM_IR_GLOSSARY)
        }
        
        task.output_data = enriched_data
        results_cache[f"{job.job_id}_metadata"] = enriched_data
    
    def _handle_pdf_generation(self, task: Task, job: AnalysisJob):
        """Generate output PDF"""
        equations = results_cache.get(f"{job.job_id}_equations", [])
        is_quantum_ir = results_cache.get(f"{job.job_id}_is_quantum_ir", False)
        keywords = results_cache.get(f"{job.job_id}_keywords", [])
        
        pdf_data = generate_equations_pdf(equations, job.filename, is_quantum_ir, keywords)
        
        if pdf_data:
            output_filename = f"analysis_{job.job_id}.pdf"
            output_path = os.path.join(ANALYSIS_RESULTS_FOLDER, output_filename)
            
            with open(output_path, 'wb') as f:
                f.write(pdf_data)
            
            task.output_data = {
                "pdf_generated": True,
                "pdf_path": output_path,
                "pdf_filename": output_filename
            }
            
            results_cache[f"{job.job_id}_pdf_path"] = output_path
        else:
            task.output_data = {"pdf_generated": False}
    
    def compile_results(self, job: AnalysisJob) -> Dict[str, Any]:
        """Compile final results and aggregate unique counts for the UI"""
        equations = results_cache.get(f"{job.job_id}_equations", [])
        is_quantum_ir = results_cache.get(f"{job.job_id}_is_quantum_ir", False)
        keywords = results_cache.get(f"{job.job_id}_keywords", [])
        metadata = results_cache.get(f"{job.job_id}_metadata", {})
        
        # Logic to aggregate unique symbols and concepts across the whole PDF
        all_unique_symbols = set()
        all_unique_concepts = set()
        
        for eq in equations:
            # Extract symbols from the equation dictionary
            if "symbols" in eq:
                all_unique_symbols.update(eq["symbols"])
            # Extract concepts from the equation dictionary
            if "concepts" in eq:
                all_unique_concepts.update(eq["concepts"])

        return {
            "job_id": job.job_id,
            "filename": job.filename,
            "status": job.status.value,
            "is_quantum_ir": is_quantum_ir,
            "keywords": keywords,
            "equations_count": len(equations),
            # These keys match what your frontend is looking for:
            "symbols": list(all_unique_symbols), 
            "concepts": list(all_unique_concepts),
            "equations": equations,
            "metadata": metadata,
            "analysis_completed_at": datetime.now().isoformat()
        }

# Initialize agent
agent = QuantumIRAgent()

# ==================== ORIGINAL UTILITY FUNCTIONS ====================

def extract_text_from_pdf(filepath):
    """Extract text from PDF"""
    text = ""
    try:
        with fitz.open(filepath) as doc:
            for page in doc:
                text += page.get_text()
        return text.lower()
    except:
        return ""

def check_quantum_ir(pdf_text):
    """Check if PDF is related to Quantum IR"""
    quantum_keywords = [
        "quantum information retrieval",
        "quantum-inspired",
        "quantum search",
        "grover's algorithm",
        "quantum ranking",
        "quantum theory",
        "qubit",
        "superposition",
        "entanglement"
    ]
    
    found_keywords = []
    for keyword in quantum_keywords:
        if keyword in pdf_text:
            found_keywords.append(keyword)
    
    is_quantum_ir = len(found_keywords) > 0
    return is_quantum_ir, found_keywords[:3]

def extract_symbols_from_equation(equation_str):
    """Extract mathematical symbols and variables"""
    clean_eq = equation_str.replace('$$', '').replace('$', '').strip()
    pattern = r'[a-zA-Z]|\\[a-zA-Z]+'
    symbols = re.findall(pattern, clean_eq)
    
    unique_symbols = []
    seen = set()
    for sym in symbols:
        if sym not in seen and sym not in ['frac', 'sqrt', 'sum', 'int', 'lim']:
            seen.add(sym)
            unique_symbols.append(sym)
    
    return unique_symbols

def generate_symbol_definitions(symbols, equation_text, context=""):
    """Generate definitions for symbols"""
    common_defs = {
        'E': 'Energy', 'm': 'Mass', 'c': 'Speed of light', 'h': 'Planck constant',
        'psi': 'Wave function', '\\psi': 'Wave function',
        'alpha': 'Probability amplitude', '\\alpha': 'Probability amplitude',
        'beta': 'Probability amplitude', '\\beta': 'Probability amplitude',
        'H': 'Hamiltonian operator', 'U': 'Unitary operator',
        'rho': 'Density matrix', '\\rho': 'Density matrix',
        'lambda': 'Eigenvalue', '\\lambda': 'Eigenvalue',
        't': 'Time', 'x': 'Position', 'p': 'Momentum', 'i': 'Imaginary unit',
        'n': 'Quantum number', 'k': 'Constant',
    }
    
    definitions = {}
    for symbol in symbols:
        definitions[symbol] = common_defs.get(symbol, f"Variable (context: {context[:30]}...)")
    
    return definitions

def identify_quantum_concepts(equation_text, explanation):
    """Identify quantum concepts"""
    concept_keywords = {
        "quantum state": ["state", "psi", "|", "⟩", "vector"],
        "superposition": ["superposition", "sum", "linear combination", "+"],
        "measurement": ["measurement", "probability", "collapse", "observe"],
        "entanglement": ["entangle", "tensor", "product", "correlation"],
        "operator": ["operator", "hamiltonian", "unitary", "hermitian"],
        "information theory": ["entropy", "information", "shannon", "bit"],
        "search algorithm": ["search", "grover", "algorithm", "query"],
        "ranking": ["rank", "score", "relevance", "similarity"],
    }
    
    text = (equation_text + " " + explanation).lower()
    concepts = []
    
    for concept, keywords in concept_keywords.items():
        for keyword in keywords:
            if keyword.lower() in text:
                concepts.append(concept)
                break
    
    return list(set(concepts))

def highlight_quantum_terms(text):
    """Highlight quantum terms"""
    highlighted_terms = []
    text_lower = text.lower()
    
    for term in QUANTUM_IR_GLOSSARY.keys():
        if term in text_lower:
            highlighted_terms.append({
                "term": term,
                "definition": QUANTUM_IR_GLOSSARY[term]
            })
    
    return highlighted_terms

def extract_equations_with_gemini(pdf_text, filename):
    """Extract equations using Gemini"""
    print(f"🤖 Calling Gemini AI to extract equations...")
    
    if GEMINI_API_KEY == "NOT_SET":
        print("❌ Gemini API key not configured.")
        return []
    
    sample_text = pdf_text[:8000]
    
    prompt = f"""EXTRACT ALL MATHEMATICAL EQUATIONS FROM THIS PDF WITH SIMPLE EXPLANATIONS:

PDF Title: {filename}

INSTRUCTIONS:
1. Find EVERY mathematical equation in the text
2. For EACH equation, provide:
   a. Provide the "equation" field in PURE LaTeX (e.g., "v = d/t" or "\\rho_{{dtv}}"). 
       DO NOT include $$ or $ delimiters in the JSON string.
   b. A VERY SIMPLE explanation in plain English that a student can understand
   c. What the equation means in everyday language
   d. Why this equation is important
3. Focus on these types of equations:
   - Quantum mechanics equations (state vectors, probabilities)
   - Information theory formulas
   - Linear algebra expressions
   - Physics equations
   - Mathematical formulas

FORMAT: Return ONLY a valid JSON array with NO additional text:
[
  {{
    "equation": "equation in LaTeX",
    "explanation": "Simple explanation for student"
  }}
]
EXAMPLES:
{{
  "equation": "|ψ⟩ = α|0⟩ + β|1⟩",
  "explanation": "This is a quantum state equation. Imagine you have a magical coin that can be both heads and tails at the same time. |ψ⟩ (pronounced 'psi') represents this magical coin state. α and β are like the amounts of 'heads-ness' and 'tails-ness'. |0⟩ means completely heads, |1⟩ means completely tails. So the coin is a mix of both!"
}}

{{
  "equation": "E = mc²",
  "explanation": "This famous equation says that energy (E) equals mass (m) times the speed of light (c) squared. Think of it like this: even a tiny bit of matter contains a HUGE amount of energy. The speed of light is very fast (300,000 km per second), and when you square it (multiply it by itself), you get an enormous number. That's why nuclear reactions release so much energy from small amounts of material."
}}

PDF TEXT:
{sample_text}
"""
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.2, "maxOutputTokens": 4000}
    }
    
    try:
        time.sleep(2)
        response = requests.post(url, json=payload, timeout=45)
        
        if response.status_code == 200:
            result = response.json()
            ai_text = result["candidates"][0]["content"]["parts"][0]["text"]
            ai_text = re.sub(r'```json\s*|\s*```', '', ai_text).strip()
            
            json_match = re.search(r'\[.*\]', ai_text, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                try:
                    equations_data = json.loads(json_str)
                    processed = []
    
                    for idx, item in enumerate(equations_data):
                        if isinstance(item, dict) and "equation" in item:
                            eq = item["equation"].strip()
                            
                            # ✅ FIX: Don't wrap in $$ if already present
                            if not (eq.startswith('$$') or eq.startswith('$')):
                                eq = f"$${eq}$$"
                            
                            # ✅ IMPORTANT: Store the equation EXACTLY as-is
                            # Don't manipulate backslashes - JSON will handle encoding
                            processed.append({
                                "id": f"eq_{idx+1}",
                                "equation": eq,  # ← Store as plain string, NOT escaped
                                "explanation": item.get("explanation", "Equation explanation")
                            })
                    print(f"✅ Extracted {len(processed)} equations")
                    return processed
                except json.JSONDecodeError:
                    return []
        else:
            print(f"❌ Gemini API error: {response.status_code}")
            return []
    except Exception as e:
        print(f"❌ Extraction failed: {e}")
        return []

def generate_equations_pdf(equations_data, filename, is_quantum_ir, keywords):
    """Generate PDF with equations"""
    try:
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=72)
        
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle('CustomTitle', parent=styles['Heading1'], fontSize=24, spaceAfter=30, alignment=TA_CENTER, textColor='#2c3e50')
        section_style = ParagraphStyle('CustomSection', parent=styles['Heading2'], fontSize=18, spaceBefore=20, spaceAfter=15, textColor='#4a6cf7')
        equation_style = ParagraphStyle('Equation', parent=styles['Code'], fontSize=14, spaceBefore=10, spaceAfter=5, leftIndent=20, textColor='#2c3e50', fontName='Courier-Bold')
        explanation_style = ParagraphStyle('Explanation', parent=styles['Normal'], fontSize=12, spaceBefore=10, spaceAfter=15, leftIndent=30, textColor='#495057')
        
        story = []
        story.append(Paragraph("Quantum IR PDF Analysis Report", title_style))
        story.append(Spacer(1, 20))
        story.append(Paragraph(f"<b>Original PDF:</b> {filename}", styles['Normal']))
        story.append(Paragraph(f"<b>Analysis Date:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
        story.append(Paragraph(f"<b>Quantum IR Detected:</b> {'Yes' if is_quantum_ir else 'No'}", styles['Normal']))
        
        if equations_data:
            story.append(Spacer(1, 30))
            story.append(Paragraph("Extracted Equations", section_style))
            
            for eq_data in equations_data:
                equation_text = eq_data.get('equation', '').replace('$$', '').replace('$', '')
                story.append(Paragraph(f"<b>{eq_data.get('id', 'Equation')}:</b>", styles['Heading3']))
                story.append(Paragraph(equation_text, equation_style))
                story.append(Paragraph(eq_data.get('explanation', ''), explanation_style))
                story.append(Spacer(1, 15))
        
        doc.build(story)
        pdf_data = buffer.getvalue()
        buffer.close()
        return pdf_data
    except Exception as e:
        print(f"❌ PDF generation failed: {e}")
        return None

# ==================== FLASK ROUTES ====================

@app.route("/", methods=["GET", "POST"])
def index():
    """Main page - handles PDF upload"""
    if request.method == "POST":
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        
        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400
        
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        # Create and start workflow
        job_id = agent.create_analysis_workflow(filename)
        
        # Execute workflow in background
        thread = threading.Thread(target=agent.execute_workflow, args=(job_id,))
        thread.daemon = True
        thread.start()
        
        return jsonify({"job_id": job_id, "status": "processing"})
    
    return render_template("main.html")

@app.route("/job/<job_id>", methods=["GET"])
def get_job_status(job_id):
    """Get job status and results"""
    job = analysis_jobs.get(job_id)
    
    if not job:
        return jsonify({"error": "Job not found"}), 404
    
    task_statuses = [
        {
            "task_id": task.task_id,
            "task_type": task.task_type.value,
            "status": task.status.value,
            "output": task.output_data,
            "error": task.error_message
        }
        for task in job.tasks
    ]
    
    return jsonify({
        "job_id": job.job_id,
        "filename": job.filename,
        "status": job.status.value,
        "tasks": task_statuses,
        "final_results": job.final_results,
        "error_log": job.error_log
    })

@app.route("/job/<job_id>/results", methods=["GET"])
def get_results(job_id):
    """Get final analysis results"""
    job = analysis_jobs.get(job_id)
    
    if not job:
        return jsonify({"error": "Job not found"}), 404
    
    if job.status != TaskStatus.COMPLETED:
        return jsonify({"error": "Analysis not completed"}), 400
    
    return jsonify(job.final_results)

@app.route("/job/<job_id>/download-pdf", methods=["GET"])
def download_pdf(job_id):
    """Download analysis PDF"""
    pdf_path = results_cache.get(f"{job_id}_pdf_path")
    
    if not pdf_path or not os.path.exists(pdf_path):
        return jsonify({"error": "PDF not found"}), 404
    
    try:
        with open(pdf_path, 'rb') as f:
            pdf_data = f.read()
        
        from flask import make_response
        response = make_response(pdf_data)
        response.headers['Content-Type'] = 'application/pdf'
        response.headers['Content-Disposition'] = f'attachment; filename=analysis_{job_id}.pdf'
        return response
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/guide")
def guide():
    """Guide page"""
    return render_template("guide.html")

@app.route("/glossary")
def get_glossary():
    """Return glossary"""
    return jsonify(QUANTUM_IR_GLOSSARY)

if __name__ == "__main__":
    print("🚀 Quantum IR PDF Analyzer")
    print(f"{'='*60}")
    
    if not os.path.exists('.env'):
        print("⚠️  WARNING: .env file not found!")
        print("   Create a file named '.env' with:")
        print("   GEMINI_API_KEY=your_actual_gemini_api_key_here")
    
    if GEMINI_API_KEY and GEMINI_API_KEY != "NOT_SET":
        print("✅ Gemini API key loaded successfully")
    else:
        print("❌ Gemini API key NOT loaded")
    
    print(f"{'='*60}")
    print("🌐 Open: http://127.0.0.1:5000")
    app.run(debug=True)
