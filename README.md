# Quantum IR PDF Analyzer - Agentic AI Edition

A sophisticated web application that leverages agentic AI to analyze PDF documents for quantum information retrieval concepts, equations, and technical content. Built with Flask, powered by Google's Gemini AI, and featuring an autonomous agent framework for comprehensive document analysis.

## 🌟 Features

### Core Analysis Capabilities
- **PDF Content Extraction**: Advanced text and metadata extraction from PDF documents
- **Quantum IR Detection**: Intelligent identification of quantum information retrieval concepts and terminology
- **Equation Analysis**: Mathematical equation extraction and processing with LaTeX support
- **Symbol Recognition**: Automated detection and analysis of mathematical symbols and notations
- **Concept Mapping**: Semantic analysis of quantum computing and information retrieval concepts
- **Term Highlighting**: Automatic highlighting of key quantum IR terms and phrases
- **Metadata Enrichment**: Enhanced document metadata with quantum-specific annotations

### Agentic Framework
- **Autonomous Workflow**: Self-managing analysis pipeline with task sequencing
- **Error Handling**: Robust retry mechanisms and error recovery
- **Progress Tracking**: Real-time status monitoring of analysis jobs
- **Scalable Architecture**: Queue-based task management for concurrent processing

### User Interface
- **Modern Web Interface**: Responsive design with particle animations and quantum-themed styling
- **Real-time Updates**: Live progress tracking during analysis
- **Interactive Results**: Detailed analysis reports with expandable sections
- **Math Rendering**: LaTeX equation display with MathJax integration
- **File Management**: Secure upload handling with automatic cleanup

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Google Gemini API key
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd quantum-ir-analyzer
   ```

2. **Install dependencies**
   ```bash
   pip install flask python-dotenv requests pymupdf reportlab
   ```

3. **Configure environment**
   ```bash
   # Create .env file
   echo "GEMINI_API_KEY=your_api_key_here" > .env
   ```

4. **Run the application**
   ```bash
   python app.py
   ```

5. **Access the web interface**
   ```
   http://localhost:5000
   ```

## 📋 Usage

1. **Upload PDF**: Select and upload a PDF document containing quantum IR content
2. **Start Analysis**: Click "Analyze Document" to initiate the agentic workflow
3. **Monitor Progress**: Watch real-time progress through the analysis pipeline
4. **View Results**: Explore detailed analysis results including:
   - Extracted quantum concepts
   - Mathematical equations
   - Symbol analysis
   - Term highlights
   - Generated PDF report

## 🏗️ Architecture

### Agent Framework
The application uses an autonomous agent (`QuantumIRAgent`) that manages a sequential workflow:

1. **PDF Extraction** → 2. **Quantum IR Check** → 3. **Equation Extraction** → 4. **Symbol Analysis** → 5. **Concept Identification** → 6. **Term Highlighting** → 7. **Metadata Enrichment** → 8. **PDF Generation**

### Task Management
- **Task Types**: Enum-based task classification
- **Status Tracking**: Real-time task status monitoring
- **Error Handling**: Automatic retry with configurable limits
- **Result Caching**: Efficient data storage and retrieval

### Data Structures
- **AnalysisJob**: Container for complete analysis workflows
- **Task**: Individual processing units with metadata
- **Results Cache**: Optimized data access patterns

## 🔧 Configuration

### Environment Variables
```env
GEMINI_API_KEY=your_gemini_api_key
GEMINI_MODEL=gemini-2.5-flash
```

### Directory Structure
```
quantum-ir-analyzer/
├── app.py                 # Main Flask application
├── templates/            # HTML templates
│   ├── main.html        # Main analysis interface
│   └── guide.html       # User guide
├── uploads/             # Uploaded PDF files
├── analysis_results/    # Generated analysis outputs
├── .env                 # Environment configuration
└── README.md           # This file
```

## 📚 Quantum IR Glossary

The application includes a comprehensive glossary of quantum information retrieval terms:

- **Qubit**: Basic unit of quantum information
- **Superposition**: Multiple states simultaneously
- **Entanglement**: Instant particle correlation
- **Hilbert Space**: Mathematical quantum state space
- **Unitary Operator**: Information-preserving operations
- **Grover's Algorithm**: Quantum search optimization
- And many more...

## 🛠️ Technical Details

### Dependencies
- **Flask**: Web framework
- **PyMuPDF (fitz)**: PDF processing
- **Google Gemini AI**: Advanced language model integration
- **ReportLab**: PDF generation
- **MathJax**: Mathematical rendering
- **Font Awesome**: UI icons

### API Integration
- **Gemini API**: Used for natural language processing and concept analysis
- **RESTful Design**: Clean API endpoints for analysis operations

### Security Features
- **Secure File Upload**: Werkzeug secure filename handling
- **Environment Variables**: Sensitive data protection
- **Input Validation**: Robust data sanitization

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

---

*Built with ❤️ for quantum information retrieval research and analysis*</content>
<parameter name="filePath">d:\Project\README.md