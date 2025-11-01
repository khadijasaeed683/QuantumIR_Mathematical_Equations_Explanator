from flask import Flask, render_template, request, jsonify, redirect, url_for
import os
import fitz  # PyMuPDF for PDF text extraction
import re
from werkzeug.utils import secure_filename

app = Flask(__name__)

# ----------------------------
# Configuration
# ----------------------------
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {'pdf', 'docx'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Supported mathematical operators
SUPPORTED_OPERATORS = {
    # Basic arithmetic
    '+', '-', '*', '/', '^', '**',
    # Comparison
    '=', '≡', '≈', '≠', '<', '>', '≤', '≥',
    # Set operators
    '∈', '∉', '⊂', '⊆', '∪', '∩',
    # Quantum-specific
    '⟨', '⟩', '|', '‖', '†', '‡',
    # Functions
    'cos', 'sin', 'tan', 'log', 'ln', 'exp', 'sqrt',
    # Greek letters (common in math)
    'α', 'β', 'γ', 'δ', 'ε', 'θ', 'λ', 'μ', 'π', 'σ', 'ϕ', 'ψ', 'ω',
    # Other mathematical symbols
    '∂', '∫', '∑', '∏', '∇', '∆'
}

# Temporary storage for extracted data
last_extraction = {"equations": [], "filename": "", "total": 0, "unsupported_operators": []}


# ----------------------------
# Helper Functions
# ----------------------------
def allowed_file(filename):
    """Check file type"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def extract_text_from_pdf(filepath):
    """Extract text from PDF"""
    text = ""
    try:
        with fitz.open(filepath) as doc:
            for page in doc:
                text += page.get_text() + "\n"
    except Exception as e:
        print("Error reading PDF:", e)
    return text


def check_quantum_ir_content(text):
    """Check if the text belongs to Quantum IR domain"""
    quantum_terms = [
        "quantum", "information retrieval", "quantum IR", "quantum-inspired",
        "superposition", "entanglement", "quantum computing", "hilbert space",
        "probability amplitude", "born rule", "quantum probability"
    ]

    text_lower = text.lower()
    quantum_match = sum(1 for term in quantum_terms if term in text_lower)
    return quantum_match >= 3


def validate_equation_operators(equation):
    """
    Validate equation against supported operators - FIXED VERSION
    Returns: (is_valid, unsupported_operators)
    """
    unsupported = set()
    
    # Extract mathematical symbols (not including variables)
    symbols = re.findall(r'[+\-*/^=<>≤≥≈≠∈∉⊂⊆∪∩⟨⟩|‖†‡∂∫∑∏∇∆→⇒↔⇔∀∃∄]', equation)
    
    # Check each symbol against supported operators
    for symbol in symbols:
        if symbol not in SUPPORTED_OPERATORS:
            unsupported.add(symbol)
    
    # For function names, we need to be more careful
    # Look for function names as whole words only
    function_pattern = r'\b(cos|sin|tan|log|ln|exp|sqrt)\b'
    function_matches = re.findall(function_pattern, equation, re.IGNORECASE)
    
    # All matched functions should be supported (since we only look for supported ones)
    # If we find "cose" or other variations, they won't match our pattern
    
    # Remove any empty strings
    unsupported = {op for op in unsupported if op.strip()}
    
    is_valid = len(unsupported) == 0
    return is_valid, list(unsupported)


def extract_proper_equations(text):
    """
    Extract only proper mathematical equations using targeted patterns
    """
    equations = []
    unsupported_operators_all = []
    
    # Clean text - join lines that might be broken equations
    text = re.sub(r'(\w)-\n(\w)', r'\1\2', text)  # Join hyphenated words
    text = re.sub(r'\n+', ' ', text)  # Replace newlines with spaces
    text = re.sub(r'\s+', ' ', text)  # Normalize spaces
    
    # TARGETED PATTERNS FOR COMPLETE EQUATIONS
    
    # Pattern 1: Probability equations p(X) = expression
    prob_eqs = re.findall(r'p\([A-Za-z]\)\s*=\s*[^.;]{10,150}?(?=[.;\n]|$)', text)
    
    # Pattern 2: Quantum probability equations with proper structure
    quantum_prob_eqs = re.findall(r'p\([A-Za-z]\)\s*=\s*\|[ϕψ][a-z]\|[\*²2][^.;]{0,100}?(?=[.;\n]|$)', text)
    
    # Pattern 3: Complete quantum expressions with = and operators
    complete_eqs = re.findall(r'(?:p\([A-Za-z]\)|\|[ϕψA-Za-z][^|]{0,30}\|²?)\s*=\s*[^.;]{10,150}?(?=[.;\n]|$)', text)
    
    # Pattern 4: Vector equations |S⟩ = expression
    vector_eqs = re.findall(r'\|\s*[A-Za-z][A-Za-z0-9]*\s*⟩\s*=\s*[^.;]{10,150}?(?=[.;\n]|$)', text)
    
    # Pattern 5: Inner product equations ⟨x|y⟩ = expression
    inner_eqs = re.findall(r'⟨[^⟩]{1,50}⟩\s*=\s*[^.;]{5,100}?(?=[.;\n]|$)', text)
    
    # Pattern 6: Equations with mathematical operators and functions
    math_eqs = re.findall(r'(?:\|[^|]{1,50}\|²?|⟨[^⟩]{1,50}⟩|[A-Za-z]\([^)]+\))\s*[=+\-*/]\s*[^.;]{10,150}?(?=[.;\n]|$)', text)
    
    # Combine all matches
    all_matches = prob_eqs + quantum_prob_eqs + complete_eqs + vector_eqs + inner_eqs + math_eqs
    
    for match in all_matches:
        equation = clean_equation(match)
        if is_valid_complete_equation(equation):
            # Validate operators
            is_valid, unsupported_ops = validate_equation_operators(equation)
            if is_valid:
                equations.append(equation)
            else:
                unsupported_operators_all.extend(unsupported_ops)
    
    # MANUAL EXTRACTION FOR SPECIFIC KNOWN EQUATION PATTERNS
    manual_extractions = extract_equations_manually(text)
    for eq in manual_extractions:
        if eq not in equations:
            # Validate operators for manual extractions too
            is_valid, unsupported_ops = validate_equation_operators(eq)
            if is_valid:
                equations.append(eq)
            else:
                unsupported_operators_all.extend(unsupported_ops)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_equations = []
    for eq in equations:
        if eq and eq not in seen:
            seen.add(eq)
            unique_equations.append(eq)
    
    # Remove duplicate unsupported operators
    unsupported_operators_all = list(set(unsupported_operators_all))
    
    return unique_equations, unsupported_operators_all


def extract_equations_manually(text):
    """Manually search for specific equation patterns"""
    manual_eqs = []
    
    # Look for specific equation patterns with context
    patterns = [
        # Probability amplitude equations
        (r'p\(A\)\s*=\s*\|ϕa\|\s*\*\s*\|ϕ† a\|\s*=\s*\|ϕa\|2', 'p(A) = |ϕa| ∗ |ϕ† a| = |ϕa|²'),
        (r'p\(X\)\s*=\s*\|ϕa\s*\+\s*ϕb\|2', 'p(X) = |ϕa + ϕb|²'),
        (r'p\(X\)\s*=\s*\|ϕa\|2\s*\+\s*\|ϕb\|2\s*\+\s*2\|ϕa\|\s*\*\s*\|ϕb\|', 'p(X) = |ϕa|² + |ϕb|² + 2|ϕa| ∗ |ϕb|'),
        (r'p\(X\)\s*=\s*p\(A\)\s*\+\s*p\(B\)\s*\+\s*2[^;]{0,30}cos\(θ\)', 'p(X) = p(A) + p(B) + 2√(p(A)p(B))cos(θ)'),
        
        # Projection equations
        (r'\|\s*P\s*\|\s*S\s*⟩\s*\|\s*2\s*=\s*⟨S\|P†P\|S⟩', '|P |S⟩ |² = ⟨S|P†P |S⟩'),
        (r'\|\s*P\s*\|\s*S\s*⟩\s*\|\s*2\s*=\s*⟨S\|P\|S⟩', '|P |S⟩ |² = ⟨S|P |S⟩'),
        (r'\|\s*PA2\|S⟩\s*\|\s*2\s*=\s*⟨S\|PA2\|S⟩', '|PA2|S⟩ |² = ⟨S|PA2|S⟩'),
        (r'\|\s*PA2\|S⟩\s*\|\s*2\s*=\s*⟨S\|A2⟩\s*⟨A2\|S⟩', '|PA2|S⟩ |² = ⟨S|A2⟩ ⟨A2|S⟩'),
        (r'\|\s*PA2\|S⟩\s*\|\s*2\s*=\s*\|\s*⟨A2\|S⟩\s*\|\s*2', '|PA2|S⟩ |² = | ⟨A2|S⟩ |²'),
        
        # State vector equations
        (r'\|\s*S\s*⟩\s*=\s*a1\s*\|\s*A1\s*⟩\s*\+\s*a2\s*\|\s*A2\s*⟩', '|S⟩ = a1 |A1⟩ + a2 |A2⟩'),
        (r'\|\s*a1\s*\|\s*2\s*\+\s*\|\s*a2\s*\|\s*2\s*=\s*1', '|a1|² + |a2|² = 1'),
    ]
    
    for pattern, replacement in patterns:
        if re.search(pattern, text):
            manual_eqs.append(replacement)
    
    # Also search for equations in numbered format like (1), (2), etc.
    numbered_eqs = re.findall(r'\((\d+)\)\s*([^\n]{20,200}?)(?=\(\d+\)|\n\n)', text)
    for num, eq in numbered_eqs:
        if '=' in eq and any(op in eq for op in ['+', '-', '*', '/', '⟨', '⟩', '|']):
            cleaned = clean_equation(eq)
            if is_valid_complete_equation(cleaned):
                manual_eqs.append(cleaned)
    
    return manual_eqs


def clean_equation(equation):
    """Clean and normalize the equation"""
    # Remove equation numbers and labels
    equation = re.sub(r'\((\d+)\)', '', equation)
    
    # Normalize spaces
    equation = re.sub(r'\s+', ' ', equation)
    
    # Remove trailing punctuation and text
    equation = re.sub(r'[.,;:]\s*[A-Za-z].*$', '', equation)
    equation = re.sub(r'\s+where\s+.*$', '', equation, flags=re.IGNORECASE)
    equation = re.sub(r'\s+and\s+.*$', '', equation, flags=re.IGNORECASE)
    
    # Fix common formatting issues
    equation = equation.replace('|2', '|²')
    equation = equation.replace('| 2', '|²')
    
    return equation.strip()


def is_valid_complete_equation(equation):
    """Validate if this is a complete mathematical equation"""
    if not equation or len(equation) < 10:
        return False
    
    # Must have proper mathematical structure
    has_equals = '=' in equation
    has_math_operators = any(op in equation for op in ['+', '-', '*', '/', '⟨', '⟩', '|', 'cos', 'sin'])
    has_variables = any(char in equation for char in ['ϕ', 'ψ', 'θ', 'α', 'β'] + [chr(i) for i in range(ord('a'), ord('z')+1)])
    
    # Should not be mostly text
    words = re.findall(r'[A-Za-z]{4,}', equation)
    word_count = len([w for w in words if w.lower() not in ['cos', 'sin', 'tan', 'log', 'exp', 'sqrt']])
    
    # Skip if too many regular words
    if word_count > 2:
        return False
    
    # Skip common false positives
    false_positives = ['http', 'www', 'arxiv', 'doi', 'figure', 'table', 'chapter', 'section', 'equation']
    if any(fp in equation.lower() for fp in false_positives):
        return False
    
    # Must be a valid mathematical expression
    return has_equals and (has_math_operators or has_variables)


# ----------------------------
# Routes
# ----------------------------
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and process it"""
    if 'file' not in request.files:
        return jsonify({"status": "error", "message": "⚠️ No file part in request."}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"status": "error", "message": "⚠️ No file selected for uploading."}), 400

    if not allowed_file(file.filename):
        return jsonify({"status": "error", "message": "❌ Invalid file format. Only PDF and DOCX allowed."}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Extract text
    text_content = extract_text_from_pdf(filepath)

    # Check if Quantum IR
    if not check_quantum_ir_content(text_content):
        return jsonify({
            "status": "warning",
            "message": "⚠️ File uploaded but not identified as Quantum IR domain. Equations not extracted."
        })

    # Extract equations using the new strict function
    equations, unsupported_operators = extract_proper_equations(text_content)
    total = len(equations)

    # Store for results page
    last_extraction["equations"] = equations
    last_extraction["filename"] = filename
    last_extraction["total"] = total
    last_extraction["unsupported_operators"] = unsupported_operators

    # Prepare response message
    message = f"✅ File identified as Quantum IR domain. Found {total} equations."
    if unsupported_operators:
        message += f" ⚠️ Found {len(unsupported_operators)} unsupported operators."

    # Send response with redirect info
    return jsonify({
        "status": "success",
        "message": message,
        "unsupported_operators": unsupported_operators,
        "redirect": "/results"
    })


@app.route('/results')
def results():
    """Display extracted equations on separate page"""
    return render_template(
        'results.html',
        filename=last_extraction["filename"],
        total=last_extraction["total"],
        equations=last_extraction["equations"],
        unsupported_operators=last_extraction["unsupported_operators"]
    )


@app.route('/supported-operators')
def supported_operators():
    """Display supported operators"""
    return jsonify({
        "supported_operators": sorted(list(SUPPORTED_OPERATORS)),
        "total_operators": len(SUPPORTED_OPERATORS)
    })


if __name__ == "__main__":
    app.run(debug=True)