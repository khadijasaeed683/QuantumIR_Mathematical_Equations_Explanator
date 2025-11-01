# sympy_parser.py - FIXED VERSION
from sympy import Eq, sympify, Symbol, I, pi, sqrt, cos, sin, exp, Function
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application
)
import re


# -------------------------------
# 🧹 STEP 1 — Fixed Cleaning Expression
# -------------------------------
def clean_for_sympy(expr: str) -> str:
    """Preprocess equation text to make it SymPy-compatible."""
    # Store original for debugging
    original = expr
    
    # First, handle quantum notation more carefully
    # Replace bras and kets with function notation
    expr = re.sub(r'\|([^⟩]+)⟩', r'ket_\1', expr)
    expr = re.sub(r'⟨([^|]+)\|', r'bra_\1', expr)
    
    # Handle combined bra-ket notation ⟨ψ|φ⟩ -> braket_psi_phi
    expr = re.sub(r'⟨([^⟩]+)⟩', r'braket_\1', expr)
    
    # Handle common mathematical symbols
    replacements = {
        '²': '**2',
        '³': '**3',
        '^': '**',
        '–': '-',
        '−': '-',
        '×': '*',
        '⊗': 'tensor_',
        '∩': 'intersect_',
        'Õ': 'Product_',  # Big Pi for product
        '√': 'sqrt',
        'α': 'alpha',
        'β': 'beta', 
        'γ': 'gamma',
        'δ': 'delta',
        'ε': 'epsilon',
        'θ': 'theta',
        'λ': 'lambda',
        'µ': 'mu',
        'ρ': 'rho',
        'σ': 'sigma',
        'φ': 'phi',
        'ϕ': 'phi',
        'ω': 'omega',
        '¯': 'bar_',
        '˜': 'tilde_',
        '∗': '*',
        '‖': 'norm_',
        '†': 'dagger_',
    }

    for old, new in replacements.items():
        expr = expr.replace(old, new)

    # Clean up whitespace around operators (but be careful not to remove too much)
    expr = re.sub(r'\s*\*\*\s*', '**', expr)
    expr = re.sub(r'\s*\*\s*', '*', expr)
    expr = re.sub(r'\s*\+\s*', '+', expr)
    expr = re.sub(r'\s*-\s*', '-', expr)
    expr = re.sub(r'\s*/\s*', '/', expr)
    expr = re.sub(r'\s*=\s*', '=', expr)
    
    # Handle absolute value notation |x| -> abs(x)
    expr = re.sub(r'\b\|([^|]+)\|', r'abs(\1)', expr)
    
    # Handle probability notation p(A|B) -> p_A_given_B
    expr = re.sub(r'p\(([^)|]+)\|([^)|]+)\)', r'p_\1_given_\2', expr)
    expr = re.sub(r'P\(([^)|]+)\|([^)|]+)\)', r'P_\1_given_\2', expr)
    
    # Handle intersections p(A∩B) -> p_A_intersect_B
    expr = re.sub(r'(\w)\(([^)]+)∩([^)]+)\)', r'\1_\2_intersect_\3', expr)
    
    # Fix spacing issues with numbers and variables
    expr = re.sub(r'(\d)\s+([a-zA-Z])', r'\1*\2', expr)  # 2 x -> 2*x
    expr = re.sub(r'([a-zA-Z])\s+(\d)', r'\1*\2', expr)  # x 2 -> x*2
    
    # Remove ONLY specific trailing annotations, not everything after "="
    # Remove text annotations that come AFTER the entire equation
    if 'In simple terms' in expr:
        expr = expr.split('In simple terms')[0].strip()
    if 'In [' in expr:
        expr = expr.split('In [')[0].strip()
    if 'When' in expr and expr.index('When') > expr.index('='):
        expr = expr.split('When')[0].strip()
    if 'which' in expr and expr.index('which') > expr.index('='):
        expr = expr.split('which')[0].strip()
    
    # Handle conditional statements more carefully
    if ' if ' in expr:
        # Only split if "if" comes after the main equation
        parts = expr.split(' if ', 1)
        if '=' in parts[0]:
            expr = parts[0].strip()
    
    expr = expr.strip()
    
    # Debug: print what happened during cleaning
    print(f"DEBUG: '{original}' -> '{expr}'")
    
    return expr


# ----------------------------------------
# ⚙️ STEP 2 — Fixed Equation Parsing
# ----------------------------------------
def parse_equation_to_sympy(equation: str) -> dict:
    """Parse a single equation string into SymPy form safely and extract symbols."""
    result = {
        "input_equation": equation.strip(),
        "status": "failed", 
        "error": None,
        "sympy": None,
        "symbols": []
    }

    try:
        eq_text = equation.strip()
        
        # Handle multi-equation lines (with multiple = signs)
        if eq_text.count('=') > 1:
            # For equations like "p(X) = |ϕa + ϕb|² = ...", take only the first part
            first_eq = eq_text.split('=', 1)[0] + '=' + eq_text.split('=', 1)[1].split('=', 1)[0]
            eq_text = first_eq.strip()
        
        # Handle conditional statements by extracting the mathematical part
        conditional_keywords = [' if ', ' when ', 'which ']
        for keyword in conditional_keywords:
            if keyword in eq_text.lower():
                parts = eq_text.split(keyword, 1)
                if '=' in parts[0]:
                    eq_text = parts[0].strip()
                break
        
        # Remove citations and annotations that come at the end
        if 'In [' in eq_text:
            eq_text = eq_text.split('In [')[0].strip()
        if 'In simple terms' in eq_text:
            eq_text = eq_text.split('In simple terms')[0].strip()
        
        if '=' not in eq_text:
            result["error"] = "No '=' found in equation after cleaning."
            return result

        lhs, rhs = eq_text.split('=', 1)
        lhs_clean = clean_for_sympy(lhs)
        rhs_clean = clean_for_sympy(rhs)
        
        # If cleaning resulted in empty strings, use original parts
        if not lhs_clean:
            lhs_clean = clean_for_sympy_simple(lhs)
        if not rhs_clean:
            rhs_clean = clean_for_sympy_simple(rhs)
            
        if not lhs_clean or not rhs_clean:
            result["error"] = f"Empty equation after cleaning: LHS='{lhs_clean}', RHS='{rhs_clean}'"
            return result

        transformations = (standard_transformations + (implicit_multiplication_application,))

        lhs_expr = parse_expr(lhs_clean, transformations=transformations)
        rhs_expr = parse_expr(rhs_clean, transformations=transformations)

        # Construct SymPy equation
        sympy_eq = Eq(lhs_expr, rhs_expr)
        result["sympy"] = sympy_eq
        result["status"] = "parsed"

        # Extract unique symbols from LHS and RHS
        symbols = list(lhs_expr.free_symbols.union(rhs_expr.free_symbols))
        symbol_data = [
            {"name": str(sym), "description": guess_symbol_meaning(str(sym))}
            for sym in symbols
        ]
        result["symbols"] = symbol_data

        return result

    except Exception as e:
        result["error"] = str(e)
        return result


def clean_for_sympy_simple(expr: str) -> str:
    """Simpler cleaning function as fallback"""
    if not expr:
        return expr
        
    # Basic replacements only
    replacements = {
        '²': '**2',
        '³': '**3', 
        '^': '**',
        '–': '-',
        '−': '-',
        '×': '*',
        '∗': '*',
    }
    
    for old, new in replacements.items():
        expr = expr.replace(old, new)
    
    # Basic quantum notation
    expr = re.sub(r'\|([^⟩]+)⟩', r'ket_\1', expr)
    expr = re.sub(r'⟨([^|]+)\|', r'bra_\1', expr)
    
    return expr.strip()


# ----------------------------------------
# 🧠 STEP 3 — Enhanced Symbol Meaning Guessing
# ----------------------------------------
def guess_symbol_meaning(symbol_name: str) -> str:
    """Try to infer or assign a basic meaning to a symbol name."""
    # Quantum mechanics specific symbols
    quantum_symbols = {
        "ket": "quantum state vector",
        "bra": "dual quantum state vector", 
        "braket": "inner product in quantum mechanics",
        "dagger": "Hermitian conjugate",
        "tensor": "tensor product",
        "phi": "wavefunction or angle",
        "psi": "wavefunction",
        "rho": "density matrix",
        "theta": "angle or parameter",
        "lambda": "wavelength or eigenvalue",
        "alpha": "coefficient or fine structure constant",
        "beta": "coefficient",
        "gamma": "relativistic factor or coefficient",
    }
    
    # Check for quantum notation first
    for qsym, meaning in quantum_symbols.items():
        if qsym in symbol_name.lower():
            return meaning
    
    # Probability and statistics
    prob_symbols = {
        "p_": "probability",
        "P_": "probability", 
        "given": "conditional probability",
        "intersect": "set intersection",
    }
    
    for psym, meaning in prob_symbols.items():
        if psym in symbol_name:
            return meaning

    greek_letters = {
        "alpha": "angle or coefficient",
        "beta": "angle or slope", 
        "gamma": "constant or photon symbol",
        "delta": "change or difference",
        "epsilon": "small quantity or error term",
        "theta": "angle",
        "lambda": "wavelength or eigenvalue",
        "mu": "mean or coefficient of friction",
        "pi": "mathematical constant (≈ 3.14159)",
        "rho": "density or resistivity",
        "sigma": "standard deviation or sum index",
        "phi": "angle or potential",
        "omega": "angular frequency",
    }

    # Match known Greek letters
    for greek, meaning in greek_letters.items():
        if greek in symbol_name.lower():
            return meaning

    # Common variable meanings
    common_vars = {
        "x": "independent variable or position",
        "y": "dependent variable or position", 
        "z": "third dimension variable or complex variable",
        "t": "time",
        "r": "radius or distance",
        "v": "velocity or potential",
        "a": "acceleration or coefficient",
        "b": "coefficient or y-intercept",
        "c": "speed of light or constant",
        "d": "distance or document (in IR)",
        "e": "Euler's number or energy",
        "f": "function or frequency",
        "g": "gravitational acceleration or function",
        "h": "height or Planck's constant",
        "i": "imaginary unit or index",
        "j": "imaginary unit or index",
        "k": "spring constant or proportionality constant",
        "l": "length or angular momentum",
        "m": "mass",
        "n": "number of terms or index",
        "p": "momentum or probability",
        "q": "charge or heat or query",
        "s": "distance or entropy",
        "E": "energy or electric field",
        "F": "force",
        "H": "Hamiltonian or enthalpy",
        "P": "power or probability",
        "Q": "charge or heat",
        "R": "resistance or radius",
        "S": "entropy or action",
        "T": "temperature or period",
        "V": "volume or potential",
        "X": "random variable or position",
    }

    # Check single letter variables first
    if len(symbol_name) == 1 and symbol_name in common_vars:
        return common_vars[symbol_name]
    
    # Check if it starts with a common variable
    first_char = symbol_name[0]
    if first_char in common_vars:
        base_meaning = common_vars[first_char]
        if symbol_name[1:].isdigit():  # Like x1, x2, etc.
            return f"{base_meaning} (indexed)"
    
    # Check for subscript notation
    if '_' in symbol_name:
        parts = symbol_name.split('_')
        if len(parts[0]) == 1 and parts[0] in common_vars:
            return f"{common_vars[parts[0]]} with subscript {parts[1]}"
    
    return "unknown or user-defined variable"


# ----------------------------------------
# 🧩 STEP 4 — Batch Parser 
# ----------------------------------------
def parse_equations_list(equations):
    """Parse a list of equations and return structured results."""
    parsed_results = []
    for eq in equations:
        if eq.strip():
            parsed_results.append(parse_equation_to_sympy(eq))
    return parsed_results


# ----------------------------------------
# 🖨️ STEP 5 — Pretty Output Helper
# ----------------------------------------
def get_equation_pretty_output(parsed_result: dict) -> str:
    """Return formatted human-readable SymPy output."""
    try:
        if parsed_result["status"] == "parsed":
            sympy_form = parsed_result["sympy"]
            symbols_info = "\n".join(
                [f"   • {s['name']}: {s['description']}" for s in parsed_result["symbols"]]
            )
            return (
                f"✅ {parsed_result['input_equation']} → {sympy_form}\n"
                f"Symbols Detected:\n{symbols_info if symbols_info else '   (none)'}"
            )
        else:
            return (
                f"❌ Could not parse: {parsed_result['input_equation']}\n"
                f"Error: {parsed_result.get('error', 'Unknown error')}"
            )
    except KeyError:
        return f"⚠️ Parsing failed for malformed result: {parsed_result}"