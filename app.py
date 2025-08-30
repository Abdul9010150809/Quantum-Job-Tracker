import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sqlite3
from datetime import datetime, timedelta
from qiskit import QuantumCircuit, transpile
from qiskit.visualization import plot_histogram
from qiskit.quantum_info import Statevector, partial_trace, entropy
import time
import json
import threading
from typing import Optional, Dict, List, Any, Tuple
import requests
import io
import base64
import warnings
import random
import sympy as sp
from scipy.optimize import minimize
warnings.filterwarnings('ignore')
from mpl_toolkits.mplot3d import Axes3D

# Optional imports with better error handling
try:
    from qiskit_aer import AerSimulator, Aer
    from qiskit.qasm3 import dumps, loads
    from qiskit.circuit.library import QFT, EfficientSU2, RealAmplitudes, TwoLocal
except ImportError:
    AerSimulator = None
    dumps = None
    loads = None

# ----------------------------
# Database with improved schema and error handling
# ----------------------------
DB_NAME = "quantum_jobs.db"

def init_db():
    """Initialize database with improved schema"""
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        
        # Drop existing table to recreate with correct schema
        cursor.execute("DROP TABLE IF EXISTS jobs")
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                ibm_job_id TEXT UNIQUE,
                status TEXT NOT NULL,
                circuit TEXT NOT NULL,
                result TEXT,
                backend_name TEXT,
                shots INTEGER DEFAULT 1024,
                execution_time REAL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                tags TEXT,
                circuit_type TEXT,
                qubits INTEGER,
                depth INTEGER
            )
        """)
        
        # Create indexes for better performance
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_status ON jobs(status)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_created_at ON jobs(created_at)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_backend ON jobs(backend_name)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_tags ON jobs(tags)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_circuit_type ON jobs(circuit_type)")
        
        # Add some sample data if the table is empty
        add_sample_data()
        
        conn.commit()
    except sqlite3.Error as e:
        st.error(f"Database error: {e}")
    finally:
        if conn:
            conn.close()

def add_sample_data():
    """Add sample data to the database"""
    sample_jobs = [
        {
            "name": "Bell State Demo",
            "ibm_job_id": "sim-1234",
            "status": "COMPLETED",
            "circuit": "OPENQASM 2.0; include \"qelib1.inc\"; qreg q[2]; creg c[2]; h q[0]; cx q[0], q[1]; measure q -> c;",
            "result": "{'00': 512, '11': 512}",
            "backend_name": "aer_simulator",
            "shots": 1024,
            "execution_time": 0.5,
            "tags": "demo,bell-state",
            "circuit_type": "bell",
            "qubits": 2,
            "depth": 2
        },
        {
            "name": "GHZ State Experiment",
            "ibm_job_id": "sim-5678",
            "status": "COMPLETED",
            "circuit": "OPENQASM 2.0; include \"qelib1.inc\"; qreg q[3]; creg c[3]; h q[0]; cx q[0], q[1]; cx q[0], q[2]; measure q -> c;",
            "result": "{'000': 510, '111': 514}",
            "backend_name": "aer_simulator",
            "shots": 1024,
            "execution_time": 0.7,
            "tags": "demo,ghz-state",
            "circuit_type": "ghz",
            "qubits": 3,
            "depth": 2
        },
        {
            "name": "Quantum Random Circuit",
            "ibm_job_id": "sim-9012",
            "status": "COMPLETED",
            "circuit": "OPENQASM 2.0; include \"qelib1.inc\"; qreg q[2]; creg c[2]; h q[0]; cx q[0], q[1]; rz(0.5) q[0]; measure q -> c;",
            "result": "{'00': 260, '01': 245, '10': 255, '11': 264}",
            "backend_name": "aer_simulator",
            "shots": 1024,
            "execution_time": 0.6,
            "tags": "random,test",
            "circuit_type": "random",
            "qubits": 2,
            "depth": 3
        },
        {
            "name": "Quantum Fourier Transform",
            "ibm_job_id": "sim-3456",
            "status": "COMPLETED",
            "circuit": "OPENQASM 2.0; include \"qelib1.inc\"; qreg q[3]; creg c[3]; h q[0]; cp(pi/2) q[0],q[1]; cp(pi/4) q[0],q[2]; h q[1]; cp(pi/2) q[1],q[2]; h q[2]; swap q[0],q[2]; measure q -> c;",
            "result": "{'000': 1024}",
            "backend_name": "aer_simulator",
            "shots": 1024,
            "execution_time": 0.8,
            "tags": "qft,demo",
            "circuit_type": "qft",
            "qubits": 3,
            "depth": 4
        }
    ]
    
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        
        # Clear any existing data
        cursor.execute("DELETE FROM jobs")
        
        for job in sample_jobs:
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cursor.execute(
                """INSERT INTO jobs 
                (name, ibm_job_id, status, circuit, result, backend_name, shots, execution_time, created_at, updated_at, tags, circuit_type, qubits, depth) 
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (job["name"], job["ibm_job_id"], job["status"], job["circuit"], job["result"], 
                 job["backend_name"], job["shots"], job["execution_time"], now, now, job["tags"],
                 job["circuit_type"], job["qubits"], job["depth"])
            )
        conn.commit()
        st.success("Sample data added successfully!")
    except sqlite3.Error as e:
        st.error(f"Failed to add sample data: {e}")
    finally:
        if conn:
            conn.close()

def add_job(name, ibm_job_id, status, circuit, result=None, backend_name=None, shots=1024, execution_time=None, tags=None, circuit_type=None, qubits=None, depth=None):
    """Add a job to the database with improved error handling"""
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute(
            """INSERT OR REPLACE INTO jobs 
            (name, ibm_job_id, status, circuit, result, backend_name, shots, execution_time, created_at, updated_at, tags, circuit_type, qubits, depth) 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (name, ibm_job_id, status, circuit, result, backend_name, shots, execution_time, now, now, tags, circuit_type, qubits, depth)
        )
        conn.commit()
        return True
    except sqlite3.Error as e:
        st.error(f"Failed to add job to database: {e}")
        return False
    finally:
        if conn:
            conn.close()

def update_job_status(job_id, status, result=None, execution_time=None):
    """Update job status with improved error handling"""
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if result is not None and execution_time is not None:
            cursor.execute(
                "UPDATE jobs SET status = ?, result = ?, execution_time = ?, updated_at = ? WHERE ibm_job_id = ?",
                (status, result, execution_time, now, job_id)
            )
        else:
            cursor.execute(
                "UPDATE jobs SET status = ?, updated_at = ? WHERE ibm_job_id = ?",
                (status, now, job_id)
            )
        conn.commit()
        return True
    except sqlite3.Error as e:
        st.error(f"Failed to update job status: {e}")
        return False
    finally:
        if conn:
            conn.close()

def get_jobs(limit=100, status_filter=None, backend_filter=None, tag_filter=None, circuit_type_filter=None):
    """Get jobs from database with filtering options"""
    try:
        conn = sqlite3.connect(DB_NAME)
        query = """
            SELECT id, name, ibm_job_id, status, circuit, result, backend_name, shots, 
                   execution_time, created_at, updated_at, tags, circuit_type, qubits, depth
            FROM jobs 
        """
        params = []
        
        if status_filter or backend_filter or tag_filter or circuit_type_filter:
            query += " WHERE "
            conditions = []
            if status_filter:
                placeholders = ','.join('?' * len(status_filter))
                conditions.append(f"status IN ({placeholders})")
                params.extend(status_filter)
            if backend_filter:
                conditions.append("backend_name LIKE ?")
                params.append(f"%{backend_filter}%")
            if tag_filter:
                conditions.append("tags LIKE ?")
                params.append(f"%{tag_filter}%")
            if circuit_type_filter:
                conditions.append("circuit_type LIKE ?")
                params.append(f"%{circuit_type_filter}%")
            query += " AND ".join(conditions)
        
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        
        cursor = conn.cursor()
        cursor.execute(query, params)
        rows = cursor.fetchall()
        return rows
    except sqlite3.Error as e:
        st.error(f"Failed to fetch jobs: {e}")
        return []
    finally:
        if conn:
            conn.close()

def get_job_stats():
    """Get statistics about jobs in database"""
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        
        # Get total jobs count
        cursor.execute("SELECT COUNT(*) FROM jobs")
        total_jobs = cursor.fetchone()[0]
        
        # Get jobs by status
        cursor.execute("SELECT status, COUNT(*) FROM jobs GROUP BY status")
        status_counts = dict(cursor.fetchall())
        
        # Get average execution time for completed jobs
        cursor.execute("SELECT AVG(execution_time) FROM jobs WHERE status = 'COMPLETED' AND execution_time IS NOT NULL")
        avg_execution_time = cursor.fetchone()[0] or 0
        
        # Get most used backend
        cursor.execute("SELECT backend_name, COUNT(*) FROM jobs WHERE backend_name IS NOT NULL GROUP by backend_name ORDER BY COUNT(*) DESC LIMIT 1")
        most_used_backend = cursor.fetchone()
        most_used_backend = most_used_backend[0] if most_used_backend else "N/A"
        
        # Get jobs per day (last 7 days)
        cursor.execute("""
            SELECT DATE(created_at) as date, COUNT(*) 
            FROM jobs 
            WHERE created_at >= DATE('now', '-7 days') 
            GROUP BY DATE(created_at) 
            ORDER by date
        """)
        jobs_per_day = dict(cursor.fetchall())
        
        # Get circuit type distribution
        cursor.execute("SELECT circuit_type, COUNT(*) FROM jobs WHERE circuit_type IS NOT NULL GROUP BY circuit_type")
        circuit_type_counts = dict(cursor.fetchall())
        
        return {
            "total_jobs": total_jobs,
            "status_counts": status_counts,
            "avg_execution_time": avg_execution_time,
            "most_used_backend": most_used_backend,
            "jobs_per_day": jobs_per_day,
            "circuit_type_counts": circuit_type_counts
        }
    except sqlite3.Error as e:
        st.error(f"Failed to get job statistics: {e}")
        return {}
    finally:
        if conn:
            conn.close()

# Initialize database
init_db()

# ----------------------------
# Enhanced Quantum Simulation
# ----------------------------
class QuantumSimulator:
    
    def __init__(self):
        self.simulators = {
            "aer_simulator": AerSimulator() if AerSimulator else None,
            "statevector_simulator": Aer.get_backend('statevector_simulator') if Aer else None,
            "unitary_simulator": Aer.get_backend('unitary_simulator') if Aer else None
        }
        
    
    def get_available_simulators(self):
        """Get available simulators"""
        return [name for name, sim in self.simulators.items() if sim is not None]
    
    def simulate_circuit(self, qc, simulator_name="aer_simulator", shots=1024):
        """Simulate a circuit using the specified simulator"""
        simulator = self.simulators.get(simulator_name)
        if not simulator:
            return None, f"Simulator {simulator_name} not available"
        
        try:
            if simulator_name == "aer_simulator":
                compiled = transpile(qc, simulator)
                job = simulator.run(compiled, shots=shots)
                result = job.result().get_counts()
                return result, None
            elif simulator_name == "statevector_simulator":
                # Create a copy without measurements for statevector simulation
                qc_copy = qc.copy()
                # Remove measurements if they exist
                if any(gate.operation.name == 'measure' for gate in qc_copy.data):
                    qc_copy.remove_final_measurements()
                compiled = transpile(qc_copy, simulator)
                job = simulator.run(compiled)
                result = job.result().get_statevector()
                return result, None
            elif simulator_name == "unitary_simulator":
                # Create a copy without measurements for unitary simulation
                qc_copy = qc.copy()
                # Remove measurements if they exist
                if any(gate.operation.name == 'measure' for gate in qc_copy.data):
                    qc_copy.remove_final_measurements()
                compiled = transpile(qc_copy, simulator)
                job = simulator.run(compiled)
                result = job.result().get_unitary()
                return result, None
        except Exception as e:
            return None, f"Simulation failed: {str(e)}"
    
    def analyze_circuit(self, qc):
        """Analyze a quantum circuit and return various metrics"""
        analysis = {}
        
        # Basic circuit info
        analysis["qubits"] = qc.num_qubits
        analysis["clbits"] = qc.num_clbits
        analysis["depth"] = qc.depth()
        analysis["size"] = qc.size()
        analysis["width"] = qc.width()
        
        # Count different gate types
        gate_counts = {}
        for instruction in qc.data:
            gate_name = instruction.operation.name
            gate_counts[gate_name] = gate_counts.get(gate_name, 0) + 1
        analysis["gate_counts"] = gate_counts
        
        # Estimate circuit complexity
        analysis["complexity"] = sum(gate_counts.values()) * analysis["qubits"]
        
        return analysis
# ----------------------------
# Qiskit helpers with improved functionality
# ----------------------------
def generate_quantum_circuit(num_qubits=2, circuit_type="bell", depth=1, params=None):
    """Generate different types of quantum circuits"""
    qc = QuantumCircuit(num_qubits, num_qubits)
    
    if circuit_type == "bell":
        qc.h(0)
        if num_qubits > 1:
            for i in range(1, num_qubits):
                qc.cx(0, i)
    elif circuit_type == "ghz":
        qc.h(0)
        if num_qubits > 1:
            for i in range(1, num_qubits):
                qc.cx(0, i)
    elif circuit_type == "random":
        for i in range(num_qubits):
            qc.h(i)
        # Add some random gates with specified depth
        for d in range(depth):
            for i in range(num_qubits):
                gate_type = random.choice(["x", "y", "z", "h", "s", "t", "sdg", "tdg", "rx", "ry", "rz"])
                if gate_type == "x":
                    qc.x(i)
                elif gate_type == "y":
                    qc.y(i)
                elif gate_type == "z":
                    qc.z(i)
                elif gate_type == "h":
                    qc.h(i)
                elif gate_type == "s":
                    qc.s(i)
                elif gate_type == "t":
                    qc.t(i)
                elif gate_type == "sdg":
                    qc.sdg(i)
                elif gate_type == "tdg":
                    qc.tdg(i)
                elif gate_type == "rx":
                    angle = random.uniform(0, 2*np.pi)
                    qc.rx(angle, i)
                elif gate_type == "ry":
                    angle = random.uniform(0, 2*np.pi)
                    qc.ry(angle, i)
                elif gate_type == "rz":
                    angle = random.uniform(0, 2*np.pi)
                    qc.rz(angle, i)
            
            # Add some entanglement
            if num_qubits > 1 and d % 2 == 0:
                for i in range(0, num_qubits-1, 2):
                    qc.cx(i, i+1)
    elif circuit_type == "qft":
        # Quantum Fourier Transform
        for j in range(num_qubits):
            qc.h(j)
            for k in range(j+1, num_qubits):
                angle = np.pi/float(2**(k-j))
                qc.cp(angle, j, k)
        for i in range(num_qubits//2):
            qc.swap(i, num_qubits-i-1)
    elif circuit_type == "variational":
        # Variational quantum circuit
        if params is None:
            params = [random.uniform(0, 2*np.pi) for _ in range(2*num_qubits)]
        
        for i in range(num_qubits):
            qc.ry(params[i], i)
            qc.rz(params[i+num_qubits], i)
        
        for i in range(num_qubits-1):
            qc.cx(i, i+1)
    elif circuit_type == "qaoa":
        # Quantum Approximate Optimization Algorithm
        if params is None:
            params = [random.uniform(0, np.pi) for _ in range(2)]
        
        # Mixer Hamiltonian
        for i in range(num_qubits):
            qc.h(i)
        
        # Cost Hamiltonian (example: MaxCut)
        for i in range(num_qubits-1):
            qc.rz(params[0], i)
            qc.cx(i, i+1)
            qc.rz(params[0], i+1)
            qc.cx(i, i+1)
        
        # Mixer again
        for i in range(num_qubits):
            qc.rx(params[1], i)
    
    qc.measure(range(num_qubits), range(num_qubits))
    return qc
# Create a global simulator instance
quantum_simulator = QuantumSimulator()

def simulate_circuit(qc, shots=1024, simulator_name="aer_simulator"):
    """Simulate a circuit using the specified simulator"""
    return quantum_simulator.simulate_circuit(qc, simulator_name, shots)

def visualize_circuit(qc):
    """Create visualization of quantum circuit using text output instead of matplotlib"""
    try:
        # Use text-based circuit drawing instead of matplotlib
        circuit_text = qc.draw(output='text')
        return circuit_text
    except Exception as e:
        st.error(f"Circuit visualization failed: {e}")
        return str(qc)

def visualize_statevector(statevector):
    """Visualize statevector"""
    try:
        # Create a simple visualization using matplotlib
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Get statevector probabilities
        probabilities = np.abs(statevector) ** 2
        n_qubits = int(np.log2(len(probabilities)))
        
        # Create labels for basis states
        labels = [format(i, '0' + str(n_qubits) + 'b') for i in range(len(probabilities))]
        
        # Plot probabilities
        ax.bar(range(len(probabilities)), probabilities)
        ax.set_xlabel('Basis State')
        ax.set_ylabel('Probability')
        ax.set_title('Statevector Probabilities')
        ax.set_xticks(range(len(probabilities)))
        ax.set_xticklabels(labels, rotation=45)
        
        plt.tight_layout()
        return fig
    except Exception as e:
        st.error(f"Statevector visualization failed: {e}")
        return None

def visualize_unitary(unitary_matrix):
    """Return 2D and 3D matplotlib figures for a unitary matrix"""
    n = unitary_matrix.shape[0]

    # --- 2D Plot ---
    fig_2d = plt.figure(figsize=(5,5))
    plt.imshow(np.abs(unitary_matrix), cmap='viridis')
    plt.colorbar(label='Magnitude')
    plt.title("Unitary Matrix (2D)")

    # --- 3D Plot ---
    fig_3d = plt.figure(figsize=(6,5))
    ax = fig_3d.add_subplot(111, projection='3d')

    X, Y = np.meshgrid(range(n), range(n))
    Z = np.abs(unitary_matrix)
    
    ax.plot_surface(X, Y, Z, cmap='viridis')
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')
    ax.set_zlabel('Magnitude')
    ax.set_title("Unitary Matrix (3D)")

    return fig_2d, fig_3d


def calculate_entanglement_entropy(statevector, subsystem_size=1):
    """Calculate entanglement entropy for a subsystem"""
    try:
        n_qubits = int(np.log2(len(statevector)))
        
        # Create reduced density matrix
        keep = list(range(subsystem_size))
        trace_out = list(range(subsystem_size, n_qubits))
        
        rho = partial_trace(statevector, trace_out)
        
        # Calculate von Neumann entropy
        ent_entropy = entropy(rho)
        
        return ent_entropy
    except Exception as e:
        st.error(f"Entanglement entropy calculation failed: {e}")
        return None

def circuit_to_qasm(qc):
    """Convert QuantumCircuit to QASM string (handles different Qiskit versions)"""
    try:
        # Try different methods to get QASM representation
        if hasattr(qc, 'qasm'):
            return qc.qasm()
        elif hasattr(qc, '__str__'):
            return str(qc)
        else:
            # Fallback: manually construct a simple QASM representation
            qasm_str = "OPENQASM 2.0;\ninclude \"qelib1.inc\";\n"
            qasm_str += f"qreg q[{qc.num_qubits}];\n"
            qasm_str += f"creg c[{qc.num_clbits}];\n"
            
            # Add basic operations (this is a simplified version)
            for instruction in qc.data:
                gate_name = instruction.operation.name
                qubits = [f"q[{q.index}]" for q in instruction.qubits]
                
                if gate_name == "h":
                    qasm_str += f"h {qubits[0]};\n"
                elif gate_name == "x":
                    qasm_str += f"x {qubits[0]};\n"
                elif gate_name == "cx":
                    qasm_str += f"cx {qubits[0]}, {qubits[1]};\n"
                # Add more gates as needed
            
            qasm_str += "measure q -> c;\n"
            return qasm_str
    except Exception as e:
        st.error(f"Failed to convert circuit to QASM: {e}")
        return f"Circuit with {qc.num_qubits} qubits"

# ----------------------------
# Quantum Algorithms
# ----------------------------
def deutsch_jozsa_algorithm(oracle_type, n_qubits=3):
    """Implement Deutsch-Jozsa algorithm"""
    qc = QuantumCircuit(n_qubits+1, n_qubits)
    
    # Initialize qubits
    qc.x(n_qubits)  # Ancilla qubit in |1> state
    for i in range(n_qubits+1):
        qc.h(i)
    
    # Apply oracle
    if oracle_type == "constant_0":
        # Do nothing (identity)
        pass
    elif oracle_type == "constant_1":
        qc.x(n_qubits)  # Flip the ancilla
    elif oracle_type == "balanced":
        # CNOT with each input qubit to ancilla
        for i in range(n_qubits):
            qc.cx(i, n_qubits)
    
    # Apply Hadamard to input qubits
    for i in range(n_qubits):
        qc.h(i)
    
    # Measure input qubits
    for i in range(n_qubits):
        qc.measure(i, i)
    
    return qc
def plot_histogram_3d(counts):
    """Create a 3D histogram visualization for measurement results"""
    # Convert counts to probabilities
    total_shots = sum(counts.values())
    states = list(counts.keys())
    probabilities = [counts[state] / total_shots for state in states]
    
    fig = plt.figure(figsize=(14, 6))
    
    # 2D histogram
    ax1 = fig.add_subplot(121)
    x_pos = np.arange(len(states))
    bars = ax1.bar(x_pos, probabilities, color='lightcoral', edgecolor='darkred', 
                  linewidth=1.5, alpha=0.8)
    
    # Add value labels
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.2f}', ha='center', va='bottom', fontweight='bold')
    
    ax1.set_xlabel('Quantum State', fontweight='bold')
    ax1.set_ylabel('Probability', fontweight='bold')
    ax1.set_title('Measurement Results (2D)', fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(states, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)
    
    # 3D histogram
    ax2 = fig.add_subplot(122, projection='3d')
    
    # Create 3D bars
    dx = dy = 0.8
    dz = probabilities
    zpos = np.zeros_like(probabilities)
    xpos = ypos = np.arange(len(probabilities))
    
    colors = plt.cm.plasma(probabilities / np.max(probabilities))
    
    ax2.bar3d(xpos, ypos, zpos, dx, dy, dz, color=colors, alpha=0.8, 
             edgecolor='darkred', linewidth=1.5)
    
    ax2.set_xlabel('State Index', fontweight='bold')
    ax2.set_ylabel('State Index', fontweight='bold')
    ax2.set_zlabel('Probability', fontweight='bold')
    ax2.set_title('Measurement Results (3D)', fontweight='bold')
    ax2.set_xticks(xpos)
    ax2.set_xticklabels(states, rotation=45, ha='right')
    
    # Adjust viewing angle
    ax2.view_init(elev=20, azim=30)
    
    plt.tight_layout()
    return fig
def grovers_algorithm(num_qubits=3, marked_element=0):
    """Implement Grover's algorithm with compatible gate operations"""
    n = num_qubits
    qc = QuantumCircuit(n, n)
    
    # Initialize qubits
    for i in range(n):
        qc.h(i)
    
    # Number of iterations (optimal for single marked element)
    iterations = int(np.pi/4 * np.sqrt(2**n))
    
    for _ in range(iterations):
        # Oracle (mark the element) - simplified version
        # Apply X gates to create the marked state pattern
        binary_rep = format(marked_element, f'0{n}b')
        for i, bit in enumerate(binary_rep):
            if bit == '0':
                qc.x(i)
        
        # Apply controlled-Z operation (simplified oracle)
        if n > 1:
            qc.h(n-1)
            qc.mcx(list(range(n-1)), n-1)  # Use mcx instead of mct
            qc.h(n-1)
        else:
            qc.z(0)
        
        # Undo X gates
        for i, bit in enumerate(binary_rep):
            if bit == '0':
                qc.x(i)
        
        # Diffusion operator
        for i in range(n):
            qc.h(i)
            qc.x(i)
        
        if n > 1:
            qc.h(n-1)
            qc.mcx(list(range(n-1)), n-1)  # Use mcx instead of mct
            qc.h(n-1)
        else:
            qc.z(0)
            
        for i in range(n):
            qc.x(i)
            qc.h(i)
    
    # Measure
    qc.measure(range(n), range(n))
    
    return qc

# ----------------------------
# AI Chatbot
# ----------------------------
class QuantumAIChatbot:
    def __init__(self):
        self.responses = {
            "hello": "Hello! I'm your Quantum Computing Assistant. How can I help you with quantum circuits or job tracking today?",
            "help": "I can help you with:\n- Creating quantum circuits\n- Understanding quantum computing concepts\n- Tracking your quantum jobs\n- Interpreting results\n\nWhat would you like to know?",
            "bell state": "A Bell state is a specific quantum state of two qubits that represents the simplest example of quantum entanglement. It's created by applying a Hadamard gate to the first qubit followed by a CNOT gate with the first qubit as control and the second as target.",
            "ghz state": "A GHZ state (Greenberger-Horne-Zeilinger) is a quantum state of three or more qubits that exhibits quantum entanglement. It's created by applying a Hadamard gate to the first qubit followed by CNOT gates between the first qubit and each of the other qubits.",
            "quantum entanglement": "Quantum entanglement is a physical phenomenon where pairs or groups of particles interact in ways such that the quantum state of each particle cannot be described independently of the state of the others, even when the particles are separated by a large distance.",
            "superposition": "Quantum superposition is a fundamental principle of quantum mechanics that states a quantum system can exist in multiple states or positions simultaneously until it is measured.",
            "quantum circuit": "A quantum circuit is a model for quantum computation where a computation is a sequence of quantum gates, which are reversible transformations on a quantum mechanical analog of an n-bit register.",
            "shots": "In quantum computing, 'shots' refers to the number of times a quantum circuit is executed to obtain measurement results. More shots provide better statistical accuracy but take longer to run.",
            "deutsch jozsa": "The Deutsch-Jozsa algorithm is a quantum algorithm that demonstrates an exponential speedup over classical algorithms for determining whether a function is constant or balanced.",
            "grovers": "Grover's algorithm is a quantum search algorithm that provides a quadratic speedup for unstructured search problems.",
            "qft": "The Quantum Fourier Transform is a linear transformation on quantum bits that is the quantum analogue of the discrete Fourier transform.",
            "variational": "Variational quantum circuits are parameterized quantum circuits used in hybrid quantum-classical algorithms for optimization and machine learning.",
            "default": "I'm not sure I understand. Could you rephrase your question? I can help with quantum computing concepts, circuit design, or job tracking."
        }
    
    def get_response(self, question):
        """Get a response to a question"""
        question_lower = question.lower()
        
        for key in self.responses:
            if key in question_lower:
                return self.responses[key]
        
        return self.responses["default"]

# Initialize AI chatbot
quantum_chatbot = QuantumAIChatbot()

# ----------------------------
# Streamlit config with improved styling
# ----------------------------
st.set_page_config(page_title="Quantum Jobs Tracker", layout="wide", page_icon="🔭")

# Simple, effective CSS fix for dark mode visibility
st.markdown("""
<style>
    /* Light mode (default) */
    :root {
        --bg-primary: #ffffff;
        --text-primary: #212529;
        --bg-secondary: #f8f9fa;
        --border-color: #dee2e6;
    }
    
    /* Dark mode specific fixes */
    [data-theme="dark"] {
        background-color: #0E1117 !important;
        color: white !important;
    }
    
    [data-theme="dark"] .stApp {
        background-color: #0E1117;
        color: white;
    }
    
    [data-theme="dark"] .stSelectbox > div > div {
        background-color: #262730 !important;
        border: 1px solid #555 !important;
        border-radius: 4px !important;
        color: white !important;
    }
    
    [data-theme="dark"] .stSelectbox > div > div:hover {
        border-color: #00ffcc !important;
    }
    
    [data-theme="dark"] .stSelectbox select {
        color: white !important;
        background-color: #262730 !important;
    }
    
    [data-theme="dark"] .stSelectbox [data-baseweb="popover"] {
        background-color: #262730 !important;
        border: 1px solid #555 !important;
    }
    
    [data-theme="dark"] .stSelectbox [data-baseweb="menu"] li {
        background-color: #262730 !important;
        color: white !important;
        padding: 8px 12px !important;
    }
    
    [data-theme="dark"] .stSelectbox [data-baseweb="menu"] li:hover {
        background-color: #00ffcc !important;
        color: black !important;
    }
    
    [data-theme="dark"] .stSelectbox label, 
    [data-theme="dark"] .stSlider label, 
    [data-theme="dark"] .stTextInput label, 
    [data-theme="dark"] .stTextArea label {
        color: white !important;
        font-weight: 600 !important;
    }
    
    [data-theme="dark"] .stTextInput>div>div>input, 
    [data-theme="dark"] .stTextArea>div>div>textarea {
        background-color: #262730 !important;
        color: white !important;
        border: 1px solid #555 !important;
        border-radius: 4px !important;
    }
    
    [data-theme="dark"] .stButton>button {
        background-color: #00ffcc !important;
        color: black !important;
        border: none !important;
        border-radius: 4px !important;
        font-weight: 600 !important;
    }
    
    [data-theme="dark"] .stSubheader, 
    [data-theme="dark"] .stHeader, 
    [data-theme="dark"] h1, 
    [data-theme="dark"] h2, 
    [data-theme="dark"] h3, 
    [data-theme="dark"] h4, 
    [data-theme="dark"] h5, 
    [data-theme="dark"] h6 {
        color: white !important;
    }
    
    [data-theme="dark"] .metric-card {
        background-color: #262730 !important;
        color: white !important;
        border: 1px solid #555 !important;
    }
    
    /* Light mode styles */
    [data-theme="light"] {
        /* Light mode will use Streamlit's default styling */
        background-color: #ffffff !important;
        color: #212529 !important;
    }
    
    [data-theme="light"] .stApp {
        background-color: #ffffff;
        color: #212529;
    }
    
    [data-theme="light"] .metric-card {
        background-color: #f8f9fa !important;
        color: #212529 !important;
        border: 1px solid #dee2e6 !important;
    }
</style>
""", unsafe_allow_html=True)

st.title("🔭 Quantum Jobs Tracker")

# ----------------------------
# Sidebar with theme switcher
# ----------------------------
with st.sidebar:
    st.header("🎨 Theme Settings")
    theme = st.selectbox("Select Theme:", ["Light", "Dark"], index=1, key="theme_select")  # Default to Dark    
    # Apply theme based on selection
    refresh_sec = st.number_input("Auto-refresh interval (seconds, 0=off)", 
         min_value=0, max_value=300, value=30, step=5, key="refresh_interval")
    
    # Data management
    st.header("🗃️ Data Management")
    if st.button("Clear All Jobs", key="clear_all_jobs"):
        if st.checkbox("I'm sure I want to clear all jobs", key="confirm_clear"):
            try:
                conn = sqlite3.connect(DB_NAME)
                cursor = conn.cursor()
                cursor.execute("DELETE FROM jobs")
                conn.commit()
                st.success("All jobs cleared")
                st.rerun()
            except sqlite3.Error as e:
                st.error(f"Failed to clear jobs: {e}")
            finally:
                if conn:
                    conn.close()
    
    # Add sample data button
    if st.button("Add Sample Data", key="add_sample_data"):
        add_sample_data()
        st.rerun()

# Apply theme
if theme == "Dark":
    st.markdown("""
    <style>
    .stApp {background: linear-gradient(135deg,#0e0f1a,#1c1e2b); color:#fff;}
    section[data-testid="stSidebar"] {background-color:#000; color:#0ff;}
    section[data-testid="stSidebar"] * {color:#0ee!important;}
    .stButton>button {background: linear-gradient(135deg,#212a3e,#394867); color:#fff; border-radius:12px; padding:0.6em 1.2em;}
    .stButton>button:hover {background: linear-gradient(135deg,#2b3450,#465a80);}
    .stDataFrame [role="grid"] td, .stDataFrame [role="grid"] th {color:#fff;background:#1e2235;border:1px solid #2f344d;}
    .stDataFrame [role="grid"] th {background:#283149;font-weight:600;}
    input, select, textarea {color:#fff;background:#2b2f45;border:1px solid #555;border-radius:10px;}
    input:focus, select:focus, textarea:focus {border:1px solid #0ff; box-shadow:0 0 8px #0ff;}
    </style>
    """, unsafe_allow_html=True)

# ----------------------------
# Dashboard with metrics
# ----------------------------
st.header("📊 Dashboard")

# Get job statistics
stats = get_job_stats()

# Display metrics
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f'<div class="metric-card">Total Jobs<br>{stats.get("total_jobs", 0)}</div>', unsafe_allow_html=True)
with col2:
    avg_time = stats.get("avg_execution_time", 0)
    st.markdown(f'<div class="metric-card">Avg Execution Time<br>{avg_time:.2f}s</div>', unsafe_allow_html=True)
with col3:
    st.markdown(
        f'<div class="metric-card">Most Used Backend<br>{stats.get("most_used_backend", "N/A")}</div>',
        unsafe_allow_html=True
    )

with col4:
    completed = stats.get("status_counts", {}).get("COMPLETED", 0)
    total = stats.get("total_jobs", 1)
    success_rate = (completed / total) * 100 if total > 0 else 0
    st.markdown(f'<div class="metric-card">Success Rate<br>{success_rate:.1f}%</div>', unsafe_allow_html=True)

# Jobs per day chart
if stats.get("jobs_per_day"):
    dates = list(stats["jobs_per_day"].keys())
    counts = list(stats["jobs_per_day"].values())
    
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(dates, counts, color='lightblue')
    ax.set_title("Jobs Submitted (Last 7 Days)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Number of Jobs")
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

def plot_histogram_3d(counts):
    """Create a 3D histogram visualization for measurement results"""
    # Convert counts to probabilities
    total_shots = sum(counts.values())
    states = list(counts.keys())
    probabilities = [counts[state] / total_shots for state in states]
    
    fig = plt.figure(figsize=(14, 6))
    
    # 2D histogram
    ax1 = fig.add_subplot(121)
    x_pos = np.arange(len(states))
    bars = ax1.bar(x_pos, probabilities, color='lightcoral', edgecolor='darkred', 
                  linewidth=1.5, alpha=0.8)
    
    # Add value labels
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.2f}', ha='center', va='bottom', fontweight='bold')
    
    ax1.set_xlabel('Quantum State', fontweight='bold')
    ax1.set_ylabel('Probability', fontweight='bold')
    ax1.set_title('Measurement Results (2D)', fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(states, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)
    
    # 3D histogram
    ax2 = fig.add_subplot(122, projection='3d')
    
    # Create 3D bars
    dx = dy = 0.8
    dz = probabilities
    zpos = np.zeros_like(probabilities)
    xpos = ypos = np.arange(len(probabilities))
    
    colors = plt.cm.plasma(probabilities / np.max(probabilities))
    
    ax2.bar3d(xpos, ypos, zpos, dx, dy, dz, color=colors, alpha=0.8, 
             edgecolor='darkred', linewidth=1.5)
    
    ax2.set_xlabel('State Index', fontweight='bold')
    ax2.set_ylabel('State Index', fontweight='bold')
    ax2.set_zlabel('Probability', fontweight='bold')
    ax2.set_title('Measurement Results (3D)', fontweight='bold')
    ax2.set_xticks(xpos)
    ax2.set_xticklabels(states, rotation=45, ha='right')
    
    # Adjust viewing angle
    ax2.view_init(elev=20, azim=30)
    
    plt.tight_layout()
    return fig

def visualize_bloch_sphere(statevector):
    """Create a Bloch sphere visualization for single qubit states"""
    try:
        from qiskit.visualization import plot_bloch_multivector
        
        fig = plot_bloch_multivector(statevector)
        return fig
    except Exception as e:
        st.error(f"Bloch sphere visualization failed: {e}")
        return None
# ----------------------------
# AI Chatbot Section
# ----------------------------
st.header("💬 Quantum AI Assistant")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display chat history
st.markdown('<div class="chat-container">', unsafe_allow_html=True)
for message in st.session_state.chat_history:
    if message["role"] == "user":
        st.markdown(f'<div class="user-message"><strong>You:</strong> {message["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="bot-message"><strong>Quantum Assistant:</strong> {message["content"]}</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Chat input with a unique key
chat_input_key = "chat_input_" + str(len(st.session_state.chat_history))
user_input = st.text_input("Ask a question about quantum computing:", key=chat_input_key)

if st.button("Send", key="send_message"):
    if user_input:
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        # Get bot response
        bot_response = quantum_chatbot.get_response(user_input)
        
        # Add bot response to chat history
        st.session_state.chat_history.append({"role": "bot", "content": bot_response})
        
        # Use a different approach to clear the input
        st.session_state.chat_input_cleared = True
        st.rerun()

if st.button("Clear Chat", key="clear_chat"):
    st.session_state.chat_history = []
    st.rerun()

# ----------------------------
# Quantum Algorithms Section
# ----------------------------
st.header("🧠 Quantum Algorithms")

algo_tab1, algo_tab2 = st.tabs(["Deutsch-Jozsa", "Grover's Search"])

with algo_tab1:
    st.subheader("Deutsch-Jozsa Algorithm")
    dj_qubits = st.slider("Number of qubits", 2, 5, 3, key="dj_qubits")
    oracle_type = st.selectbox("Oracle type", ["constant_0", "constant_1", "balanced"], key="oracle_type")
    
    if st.button("Run Deutsch-Jozsa", key="run_dj"):
        with st.spinner("Running Deutsch-Jozsa algorithm..."):
            qc = deutsch_jozsa_algorithm(oracle_type, dj_qubits-1)  # n-1 input qubits, 1 ancilla
            result, error = simulate_circuit(qc, shots=1024)
            
            if result:
                st.success("Deutsch-Jozsa algorithm completed!")
                
                # Display circuit
                st.subheader("Circuit")
                circuit_text = visualize_circuit(qc)
                st.text(circuit_text)
                
                # Display results
                st.subheader("Results")
                fig = plot_histogram(result)
                st.pyplot(fig)
                
                # Interpretation
                if oracle_type in ["constant_0", "constant_1"]:
                    st.info("The function is constant (all zeros in measurement)")
                else:
                    st.info("The function is balanced (not all zeros in measurement)")
            else:
                st.error(f"Deutsch-Jozsa algorithm failed: {error}")

with algo_tab2:
    st.subheader("Grover's Search Algorithm")
    grover_qubits = st.slider("Number of qubits", 2, 4, 3, key="grover_qubits")
    marked_element = st.number_input("Marked element", 0, 2**grover_qubits-1, 0, key="marked_element")
    
    if st.button("Run Grover's Search", key="run_grover"):
        with st.spinner("Running Grover's search algorithm..."):
            qc = grovers_algorithm(grover_qubits, marked_element)
            result, error = simulate_circuit(qc, shots=1024)
            
            if result:
                st.success("Grover's search algorithm completed!")
                
                # Display circuit
                st.subheader("Circuit")
                circuit_text = visualize_circuit(qc)
                st.text(circuit_text)
                
                # Display results
                st.subheader("Results")
                fig = plot_histogram(result)
                st.pyplot(fig)
                
                # Find the most probable result
                most_probable = max(result, key=result.get)
                st.info(f"Most probable result: {most_probable} (marked element: {format(marked_element, '0' + str(grover_qubits) + 'b')})")
            else:
                st.error(f"Grover's search algorithm failed: {error}")

# ----------------------------
# Circuit Designer
# ----------------------------
st.header("🎛️ Circuit Designer")

circuit_tab1, circuit_tab2, circuit_tab3 = st.tabs(["Preset Circuits", "Custom Circuit", "Advanced Simulation"])

with circuit_tab1:
    st.subheader("Preset Quantum Circuits")
    
    preset_col1, preset_col2 = st.columns(2)
    with preset_col1:
        circuit_type = st.selectbox("Circuit Type", 
                                  ["bell", "ghz", "random", "qft", "variational", "qaoa"],
                                  help="Select a type of quantum circuit to generate", key="circuit_type_select")
        num_qubits = st.slider("Number of Qubits", 2, 10, 2, key="num_qubits_slider")
        
        if circuit_type == "random":
            depth = st.slider("Circuit Depth", 1, 10, 3, key="circuit_depth_slider")
        else:
            depth = 1
    
    with preset_col2:
        shots = st.slider("Shots", 100, 10000, 1024, key="shots_slider")
        job_name = st.text_input("Job Name", value=f"{circuit_type}-{num_qubits}q", key="job_name_input")
        tags = st.text_input("Tags (comma-separated)", help="Add tags to organize your jobs", key="tags_input")
    
    if st.button("Generate Circuit", key="generate_circuit_button"):
        with st.spinner("Generating circuit..."):
            qc = generate_quantum_circuit(num_qubits, circuit_type, depth)
            
            # Display circuit
            st.subheader("Circuit Visualization")
            circuit_text = visualize_circuit(qc)
            st.text(circuit_text)
            
            # Circuit analysis
            analysis = quantum_simulator.analyze_circuit(qc)
            st.subheader("Circuit Analysis")
            st.markdown(f"""
            <div class="circuit-analysis">
                <strong>Qubits:</strong> {analysis['qubits']}<br>
                <strong>Depth:</strong> {analysis['depth']}<br>
                <strong>Size:</strong> {analysis['size']}<br>
                <strong>Width:</strong> {analysis['width']}<br>
                <strong>Complexity:</strong> {analysis['complexity']}<br>
                <strong>Gate Counts:</strong> {', '.join([f'{k}: {v}' for k, v in analysis['gate_counts'].items()])}
            </div>
            """, unsafe_allow_html=True)
            
            # Display statevector (for small circuits)
            if num_qubits <= 5:
                st.subheader("Statevector Visualization")
                state_result, error = simulate_circuit(qc, simulator_name="statevector_simulator")
                if state_result is not None:
                    state_fig = visualize_statevector(state_result)
                    if state_fig:
                        st.pyplot(state_fig)
                    
                    # Calculate entanglement entropy
                    entropy = calculate_entanglement_entropy(state_result, 1)
                    if entropy is not None:
                        st.info(f"Entanglement entropy: {entropy:.4f}")
            
            # Store circuit in session state for execution
            st.session_state.generated_circuit = qc
            st.session_state.circuit_name = job_name
            st.session_state.circuit_tags = tags
            st.session_state.circuit_shots = shots
            st.session_state.circuit_type = circuit_type
            st.session_state.circuit_qubits = num_qubits
            st.session_state.circuit_depth = depth
            
            st.success("Circuit generated successfully!")

with circuit_tab2:
    st.subheader("Custom Quantum Circuit")
    
    # QASM input
    qasm_code = st.text_area("Enter QASM Code", height=200,
                            value="""OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
h q[0];
cx q[0], q[1];
measure q -> c;""", key="qasm_text_area")
    
    custom_shots = st.slider("Shots", 100, 10000, 1024, key="custom_shots_slider")
    custom_job_name = st.text_input("Job Name", value="custom-circuit", key="custom_job_name_input")
    custom_tags = st.text_input("Tags (comma-separated)", key="custom_tags_input", help="Add tags to organize your jobs")
    
    if st.button("Load Circuit", key="load_circuit_button"):
        try:
            # Use QuantumCircuit.from_qasm_str() instead of undefined 'loads'
            qc = QuantumCircuit.from_qasm_str(qasm_code)
            st.session_state.generated_circuit = qc
            st.session_state.circuit_name = custom_job_name
            st.session_state.circuit_tags = custom_tags
            st.session_state.circuit_shots = custom_shots
            st.session_state.circuit_type = "custom"
            st.session_state.circuit_qubits = qc.num_qubits
            st.session_state.circuit_depth = qc.depth()
            
            # Display circuit
            st.subheader("Circuit Visualization")
            circuit_text = visualize_circuit(qc)
            st.text(circuit_text)
            
            # Circuit analysis
            analysis = quantum_simulator.analyze_circuit(qc)
            st.subheader("Circuit Analysis")
            st.markdown(f"""
            <div class="circuit-analysis">
                <strong>Qubits:</strong> {analysis['qubits']}<br>
                <strong>Depth:</strong> {analysis['depth']}<br>
                <strong>Size:</strong> {analysis['size']}<br>
                <strong>Width:</strong> {analysis['width']}<br>
                <strong>Complexity:</strong> {analysis['complexity']}<br>
                <strong>Gate Counts:</strong> {', '.join([f'{k}: {v}' for k, v in analysis['gate_counts'].items()])}
            </div>
            """, unsafe_allow_html=True)
            
            st.success("Circuit loaded successfully!")
            
        except Exception as e:
            st.error(f"Failed to load circuit: {e}")
            # Add more detailed error information
            import traceback
            st.text(traceback.format_exc())

with circuit_tab3:
    st.subheader("Advanced Simulation Options")
    
    if hasattr(st.session_state, 'generated_circuit'):
        simulator_options = quantum_simulator.get_available_simulators()
        selected_simulator = st.selectbox("Select Simulator", simulator_options, key="simulator_select")
        
        if selected_simulator == "statevector_simulator":
            if st.button("Run Statevector Simulation", key="run_statevector_button"):
                with st.spinner("Running statevector simulation..."):
                    result, error = simulate_circuit(st.session_state.generated_circuit, simulator_name=selected_simulator)
                    
                    if result is not None:
                        st.success("Statevector simulation completed!")
                        
                        # Display statevector
                        st.subheader("Statevector")
                        fig = visualize_statevector(result)
                        if fig:
                            st.pyplot(fig)
                        
                        # Display Bloch sphere for single qubit
                        if st.session_state.generated_circuit.num_qubits == 1:
                            st.subheader("Bloch Sphere Representation")
                            bloch_fig = visualize_bloch_sphere(result)
                            if bloch_fig:
                                st.pyplot(bloch_fig)
                        
                        # Calculate and display entanglement entropy
                        try:
                            # Extract statevector properly
                            if hasattr(result, 'get_statevector'):
                                statevector = result.get_statevector()
                            elif hasattr(result, 'data'):
                                # For newer Qiskit versions, check if result.data is a method or attribute
                                if callable(result.data):
                                    statevector_data = result.data()
                                else:
                                    statevector_data = result.data
                                
                                if hasattr(statevector_data, 'get'):
                                    statevector = statevector_data.get('statevector', None)
                                else:
                                    # If it's already a numpy array (which is the case for statevector results)
                                    statevector = statevector_data
                            else:
                                # If it's already a statevector numpy array
                                statevector = result
                            
                            if statevector is not None:
                                # Debug info
                                st.write(f"Statevector type: {type(statevector)}")
                                st.write(f"Statevector shape: {statevector.shape if hasattr(statevector, 'shape') else 'N/A'}")
                                
                                for i in range(st.session_state.generated_circuit.num_qubits):
                                    entropy_val = calculate_entanglement_entropy(statevector, i)
                                    if entropy_val is not None:
                                        st.info(f"Entanglement entropy for qubit {i}: {entropy_val:.4f}")
                            else:
                                st.warning("Could not extract statevector from result")
                                
                        except Exception as e:
                            st.error(f"Error calculating entanglement entropy: {str(e)}")
                            import traceback
                            st.text(traceback.format_exc())
                            
                    else:
                        st.error(f"Statevector simulation failed: {error}")
        
        elif selected_simulator == "unitary_simulator":
            if st.button("Run Unitary Simulation", key="run_unitary_button"):
                with st.spinner("Running unitary simulation..."):
                    result, error = simulate_circuit(st.session_state.generated_circuit, simulator_name=selected_simulator)
                    
                    if result is not None:
                        st.success("Unitary simulation completed!")
                        
                        # Display both 2D and 3D figures
                        st.subheader("Unitary Matrix (2D & 3D)")
                        try:
                            fig_2d, fig_3d = visualize_unitary(result)
                            if fig_2d:
                                st.pyplot(fig_2d)
                            if fig_3d:
                                st.pyplot(fig_3d)
                        except Exception as e:
                            st.error(f"Error visualizing unitary matrix: {str(e)}")
                    else:
                        st.error(f"Unitary simulation failed: {error}")
# ----------------------------
# ----------------------------
# Job Execution
# ----------------------------
if hasattr(st.session_state, 'generated_circuit'):
    st.header("⚡ Execute Circuit")
    
    exec_col1, exec_col2 = st.columns(2)
    
    with exec_col1:
        st.write("**Circuit ready for execution**")
        st.write(f"Qubits: {st.session_state.generated_circuit.num_qubits}")
        st.write(f"Depth: {st.session_state.generated_circuit.depth()}")
        st.write(f"Shots: {st.session_state.circuit_shots}")
    
    with exec_col2:
        simulator_options = quantum_simulator.get_available_simulators()
        selected_simulator = st.selectbox("Select Simulator", simulator_options, key="execution_simulator_select")
        
        if st.button("Execute Circuit", type="primary", key="execute_circuit_button"):
            with st.spinner("Executing circuit..."):
                qc = st.session_state.generated_circuit
                
                result, error = simulate_circuit(qc, st.session_state.circuit_shots, selected_simulator)
                if result:
                    job_id = f"sim-{random.randint(1000, 9999)}"
                    backend_name = selected_simulator
                    
                    # Convert circuit to QASM string
                    circuit_qasm = circuit_to_qasm(qc)
                    
                    add_job(
                        st.session_state.circuit_name, job_id, "COMPLETED", 
                        circuit_qasm, str(result), backend_name, 
                        st.session_state.circuit_shots, 0, st.session_state.circuit_tags,
                        st.session_state.circuit_type, st.session_state.circuit_qubits, st.session_state.circuit_depth
                    )
                    
                    st.success(f"Simulation completed on {backend_name}")
                    
                    # Display results
                    st.subheader("Results")
                    
                    if selected_simulator == "aer_simulator":
                        fig = plot_histogram(result)  # Use Qiskit's built-in function
                        st.pyplot(fig)
                        
                        # Calculate statistics
                        total_shots = sum(result.values())
                        most_probable = max(result, key=result.get)
                        probability = result[most_probable] / total_shots
                        
                        st.info(f"Most probable state: {most_probable} ({probability:.2%})")
                    
                    elif selected_simulator == "statevector_simulator":
                        fig = visualize_statevector(result)
                        if fig:
                            st.pyplot(fig)
                        
                        # Display Bloch sphere for single qubit
                        if st.session_state.generated_circuit.num_qubits == 1:
                            st.subheader("Bloch Sphere Representation")
                            bloch_fig = visualize_bloch_sphere(result)
                            if bloch_fig:
                                st.pyplot(bloch_fig)
                    
                    elif selected_simulator == "unitary_simulator":
                        fig_2d, fig_3d = visualize_unitary(result)
                        if fig_2d:
                            st.pyplot(fig_2d)
                        if fig_3d:
                            st.pyplot(fig_3d)
                    
                    st.rerun()
                else:
                    st.error(f"Simulation failed: {error}")
# ----------------------------
# Job Management
# ----------------------------
st.header("📋 Job Management")

# Filter options
filter_col1, filter_col2, filter_col3, filter_col4 = st.columns(4)
with filter_col1:
    status_filter = st.multiselect("Filter by status", ["QUEUED", "RUNNING", "COMPLETED", "ERROR", "CANCELLED"], key="status_filter")
with filter_col2:
    backend_filter = st.text_input("Backend contains", key="backend_filter")
with filter_col3:
    tag_filter = st.text_input("Tag contains", key="tag_filter")
with filter_col4:
    circuit_type_filter = st.text_input("Circuit type contains", key="circuit_type_filter")
    limit = st.slider("Jobs to show", 5, 200, 20, key="limit_slider")

# Get filtered jobs
rows = get_jobs(limit=limit, 
                status_filter=status_filter if status_filter else None, 
                backend_filter=backend_filter if backend_filter else None,
                tag_filter=tag_filter if tag_filter else None,
                circuit_type_filter=circuit_type_filter if circuit_type_filter else None)

if rows:
    df_db = pd.DataFrame(rows, columns=[
        "ID", "Name", "IBM Job ID", "Status", "Circuit", "Result", 
        "Backend", "Shots", "Execution Time", "Created At", "Updated At", "Tags",
        "Circuit Type", "Qubits", "Depth"
    ])
    
    # Format status with badges
    def format_status(status):
        return f'<span class="status-badge {status.lower()}">{status}</span>'
    
    df_display = df_db.copy()
    df_display["Status"] = df_display["Status"].apply(format_status)
    
    # Display jobs table
    st.markdown(df_display[["Name", "IBM Job ID", "Status", "Backend", "Shots", "Execution Time", "Created At", "Tags", "Circuit Type", "Qubits", "Depth"]].to_html(escape=False), unsafe_allow_html=True)
    
    # Job details
    if not df_db.empty:
        job_options = [f"{row['Name']} ({row['IBM Job ID']}) - {row['Status']}" for _, row in df_db.iterrows()]
        selected_job = st.selectbox("Select job to view details", job_options, key="job_select")
        
        if selected_job:
            job_id = selected_job.split("(")[-1].split(")")[0]
            job_data = df_db[df_db["IBM Job ID"] == job_id].iloc[0]
            
            with st.expander("Job Details", expanded=True):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Name:** {job_data['Name']}")
                    st.write(f"**Status:** {job_data['Status']}")
                    st.write(f"**Backend:** {job_data['Backend']}")
                    st.write(f"**Shots:** {job_data['Shots']}")
                    st.write(f"**Circuit Type:** {job_data['Circuit Type']}")
                    st.write(f"**Qubits:** {job_data['Qubits']}")
                    st.write(f"**Depth:** {job_data['Depth']}")
                    if job_data['Tags']:
                        st.write(f"**Tags:** {job_data['Tags']}")
                with col2:
                    st.write(f"**Execution Time:** {job_data['Execution Time']}s" if job_data['Execution Time'] else "**Execution Time:** N/A")
                    st.write(f"**Created:** {job_data['Created At']}")
                    st.write(f"**Updated:** {job_data['Updated At']}")
                    if job_data['IBM Job ID'].startswith('sim-'):
                        st.write("**Type:** Simulation")
                    else:
                        st.write("**Type:** IBM Quantum Hardware")
                
                # Display circuit
                st.write("**Circuit:**")
                st.code(job_data['Circuit'])
                
                # Display results if available
                if job_data['Result'] and job_data['Result'] != 'None':
                    st.write("**Result:**")
                    try:
                        result_data = eval(job_data['Result'])
                        if isinstance(result_data, dict):
                            # Measurement results
                            fig = plot_histogram(result_data)
                            st.pyplot(fig)
                        else:
                            # Statevector or unitary results
                            st.text(str(result_data)[:500] + "..." if len(str(result_data)) > 500 else str(result_data))
                    except:
                        st.text(job_data['Result'])
                
                # Action buttons
                if st.button("Delete Job", key=f"delete_{job_id}"):
                    try:
                        conn = sqlite3.connect(DB_NAME)
                        cursor = conn.cursor()
                        cursor.execute("DELETE FROM jobs WHERE ibm_job_id = ?", (job_id,))
                        conn.commit()
                        st.success("Job deleted")
                        st.rerun()
                    except sqlite3.Error as e:
                        st.error(f"Failed to delete job: {e}")
                    finally:
                        if conn:
                            conn.close()
else:
    st.info("No jobs found matching your filters.")

# ----------------------------
# Analytics and Visualization
# ----------------------------
st.header("📈 Analytics")

if rows:
    # Status distribution
    status_counts = df_db["Status"].value_counts()
    
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(8, 8))
        colors = ["#FFA500", "#87CEEB", "#90EE90", "#FF6347", "#888888"]
        
        wedges, texts, autotexts = ax.pie(
            status_counts.values, 
            labels=status_counts.index, 
            autopct='%1.1f%%',
            colors=colors[:len(status_counts)],
            startangle=90
        )
        
        ax.set_title("Job Status Distribution")
        plt.tight_layout()
        st.pyplot(fig)
    
    with col2:
        # Execution time distribution for completed jobs
        completed_jobs = df_db[df_db["Status"] == "COMPLETED"]
        if not completed_jobs.empty and completed_jobs["Execution Time"].notna().any():
            fig, ax = plt.subplots(figsize=(8, 6))
            
            ax.hist(
                completed_jobs["Execution Time"].dropna(),
                bins=10,
                color='lightblue',
                edgecolor='black'
            )
            
            ax.set_title("Execution Time Distribution (Completed Jobs)")
            ax.set_xlabel("Execution Time (s)")
            ax.set_ylabel("Count")
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.info("No execution time data available for completed jobs")
    
    # Backend usage
    backend_counts = df_db["Backend"].value_counts()
    if not backend_counts.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.bar(
            backend_counts.index,
            backend_counts.values,
            color='lightgreen',
            edgecolor='black'
        )
        
        ax.set_title("Backend Usage")
        ax.set_xlabel("Backend")
        ax.set_ylabel("Count")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig)
    
    # Circuit type distribution
    circuit_type_counts = df_db["Circuit Type"].value_counts()
    if not circuit_type_counts.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.bar(
            circuit_type_counts.index,
            circuit_type_counts.values,
            color='lightcoral',
            edgecolor='black'
        )
        
        ax.set_title("Circuit Type Distribution")
        ax.set_xlabel("Circuit Type")
        ax.set_ylabel("Count")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig)

# ----------------------------
# Auto-refresh
# ----------------------------
if refresh_sec > 0:
    time.sleep(refresh_sec)
    st.rerun()