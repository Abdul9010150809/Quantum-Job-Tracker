# Quantum Job Tracker 🌌

![Streamlit Badge](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)
![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)
![Qiskit ≥1.0.0](https://img.shields.io/badge/Qiskit-%E2%89%A51.0.0-6133BD.svg)

A comprehensive quantum computing job tracking and management platform developed for the Quantum Valley Internal Hackathon. This application provides real-time monitoring, analytics, and management of quantum computing jobs with advanced visualization capabilities.

---

## 🏆 Hackathon Project

- **Event:** Quantum Valley Internal Hackathon  
- **Team:** Team Spirit 
- **Track:** Quantum Computing & Job Management  
- **Focus:** Real-time job monitoring, analytics, and quantum algorithm integration

---

## ✨ Features

### 📊 Job Management Dashboard
- Real-time job status monitoring with comprehensive metrics
- Interactive filtering by status, backend, tags, and circuit type
- Detailed job analytics with execution time distribution
- Backend usage statistics and performance metrics

### ⚡ Quantum Algorithm Integration
- Built-in quantum algorithms: Deutsch-Jozsa, Grover's Search, and more
- Interactive circuit designer with preset and custom circuits
- Advanced simulation options with multiple simulator backends
- Real-time circuit execution and results visualization

### 🔍 Advanced Analytics
- Job status distribution charts
- Execution time analysis with statistical visualization
- Backend performance comparison across different quantum simulators
- Circuit type distribution and usage patterns

### 🎨 User Experience
- Theme customization (Light/Dark mode support)
- Auto-refresh functionality for real-time updates
- Sample data generation for testing and demonstration
- Responsive design optimized for all devices

---

## 🚀 Live Deployment

Access the production application at:  
[https://quantum-job-tracker.streamlit.app/](https://quantum-job-tracker.streamlit.app/)

---

## 📊 Application Overview

**Dashboard Metrics**
- **Total Jobs:** Comprehensive count of all quantum jobs
- **Average Execution Time:** Performance benchmarking (currently 0.65s)
- **Most Used Backend:** Popular quantum simulators (aer_simulator)
- **Success Rate:** Job completion metrics (100% success rate)

**Supported Quantum Backends**
- `aer_simulator` - Primary quantum circuit simulator
- `unitary_simulator` - Advanced unitary matrix simulator
- Additional IBM Quantum backends supported

**Circuit Types Supported**
- QFT: Quantum Fourier Transform circuits
- Random: Quantum random circuit generation
- GHZ: Greenberger-Horne-Zeilinger state experiments
- Bell: Bell state demonstrations and tests
- Custom: User-defined quantum circuits

---

## 🛠️ Technical Implementation

**Core Technologies**
- Python 3.8+
- Streamlit >=1.35.0
- Qiskit >=1.0.0
- Qiskit Aer >=0.15.1
- Qiskit IBM Runtime >=0.11.0
- Pandas >=2.0.0
- Numpy >=1.24.0
- Matplotlib >=3.7.0
- Plotly >=5.11.0
- Sympy >=1.11.0
- python-dotenv >=0.19.0

---

## 📁 Project Structure

```
Quantum-Job-Tracker/
│
├── app.py                 # Main Streamlit application
├── ibm_client.py          # IBM Quantum backend integration
│
├── requirements.txt       # Python dependencies
├── README.md              # Documentation
├── .gitignore             # Ignore rules
│
├── data/                  # Databases & sample data (ignored in git)
│   ├── jobs.db
│   └── quantum_jobs.db
│
├── output/                # Demo & docs (can keep for showcase)
│   ├── quantum-job-tracker.mp4
│   └── Quantum-job-tracker.pdf
│
├── wheels/                # Optional: prebuilt Python wheels
│
├── .venv/                 # Virtual environment (ignored)
├── .dist/                 # Build/distribution files (ignored)
└── __pycache__/           # Python cache files (ignored)

```

---

## 🚀 Quick Start

### Local Development

**Clone and setup**
```sh
git clone https://github.com/yourusername/quantum-job-tracker.git
cd quantum-job-tracker
python -m venv quantum_env
source quantum_env/bin/activate  # Linux/Mac
# or
quantum_env\Scripts\activate    # Windows
```

**Install dependencies**
```sh
pip install -r requirements.txt
```

**Run the application**
```sh
streamlit run app.py
```

**Access the application**  
Open your browser to [http://localhost:8501](http://localhost:8501)

---

## ⚙️ Configuration

**Environment Setup Example**
```python
AUTO_REFRESH_INTERVAL = 30  # seconds
DEFAULT_BACKEND = "aer_simulator"
DEFAULT_SHOTS = 1024
THEME_OPTIONS = ["Light", "Dark", "Auto"]
```

**Supported Quantum Backends**
- `aer_simulator` - High-performance quantum circuit simulator
- `unitary_simulator` - Unitary matrix computation
- `statevector_simulator` - Statevector simulation
- IBM Quantum backends (with appropriate credentials)

---

## 🎮 Usage Examples

**Running Grover's Search Algorithm**
```python
# Configure Grover's algorithm
qubits = 3
marked_element = 5
shots = 1024

# Execute and get results
results = run_grover_search(qubits, marked_element, shots)
```

**Creating Custom Circuits**
```python
# Build a custom quantum circuit
qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)
qc.measure_all()

# Execute on selected backend
job = execute_circuit(qc, backend="aer_simulator", shots=1024)
```

---

## 📊 Data Models

**Job Information**
```json
{
    "name": "Quantum Fourier Transform",
    "job_id": "sim-3456",
    "status": "COMPLETED",
    "backend": "aer_simulator",
    "shots": 1024,
    "execution_time": 0.8,
    "created_at": "2025-08-30 22:55:12",
    "tags": ["qft_demo"],
    "circuit_type": "qft",
    "qubits": 3
}
```

**Analytics Data**
```json
{
    "total_jobs": 15,
    "avg_execution_time": 0.65,
    "most_used_backend": "aer_simulator",
    "success_rate": 100.0,
    "job_distribution": {
        "COMPLETED": 12,
        "RUNNING": 2,
        "QUEUED": 1
    }
}
```

---

## 🔧 Development Guide

- **Adding New Quantum Algorithms**
  - Create algorithm implementation in `src/quantum/algorithms/`
  - Add UI component in `src/components/algorithms.py`
  - Update routing in `app.py`

- **Extending Visualization**
  - Add new chart types in `src/utils/visualization.py`
  - Create corresponding UI components
  - Integrate with data management system

---

## 🚀 Deployment

### Streamlit Cloud Deployment
- Push code to GitHub repository
- Connect at [share.streamlit.io](https://share.streamlit.io)
- Deploy from main branch
- Configure environment variables as needed

### Custom Deployment
The application can be deployed on:
- Streamlit Sharing (recommended)
- Heroku with Docker container
- AWS EC2 with reverse proxy
- Google Cloud Run (serverless)

---

## 📈 Performance Metrics

- **Dashboard Load Time:** < 1.5 seconds
- **Circuit Execution:** < 2 seconds for typical circuits
- **Data Processing:** Real-time analytics updates
- **Memory Usage:** Optimized for efficient operation

---


## 🙏 Acknowledgments

- Quantum Valley for hosting the hackathon
- Streamlit for the excellent web framework
- Qiskit team for quantum computing resources
- IBM Quantum for simulator technologies

---

## 📞 Support

For questions or support:
- **GitHub Issues:** Create an issue
- **Email:** samm41236@gmail.com
- **Documentation:** Full documentation available in `/output` folder

---

Built with ❤️ for the quantum computing community

> "Managing quantum jobs
