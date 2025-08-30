# ibm_client.py
from datetime import datetime, timedelta
import random

from qiskit import IBMQ, transpile, Aer, execute
from qiskit.providers.ibmq import least_busy # type: ignore

# Initialize IBM provider
def get_provider():
    IBMQ.load_account()  # make sure you did `IBMQ.save_account("YOUR_API_KEY")`
    provider = IBMQ.get_provider(hub='ibm-q')
    return provider

def submit_circuit(qc):
    provider = get_provider()
    backend = least_busy(provider.backends(filters=lambda b: b.configuration().n_qubits >= qc.num_qubits 
                                           and not b.configuration().simulator 
                                           and b.status().operational==True))
    transpiled = transpile(qc, backend)
    job = backend.run(transpiled)
    return job.job_id(), backend.name()

def get_job_result(job_id):
    provider = get_provider()
    job = provider.backends()[0].retrieve_job(job_id)
    result = job.result()
    counts = result.get_counts()
    return str(counts)

try:
    from qiskit_ibm_runtime import QiskitRuntimeService # type: ignore
    IBM_AVAILABLE = True
except Exception:
    IBM_AVAILABLE = False

# ---------- MOCK DATA ----------
def _mock_jobs(n=10):
    statuses = ["QUEUED", "RUNNING", "COMPLETED", "ERROR"]
    backends = ["ibmq_qasm_simulator", "ibmq_manila", "ibm_perth", "ibmq_belem"]
    jobs = []
    now = datetime.utcnow()
    for i in range(n):
        jid = f"mock-{1000+i}"
        backend = random.choice(backends)
        status = random.choices(statuses, weights=[0.2,0.2,0.5,0.1])[0]
        created = now - timedelta(minutes=random.randint(1, 720))
        finished = created + timedelta(minutes=random.randint(1,60)) if status in ["COMPLETED","ERROR"] else None
        shots = random.choice([1024,2048,4096])
        jobs.append({
            "job_id": jid,
            "backend_name": backend,
            "status": status,
            "created": created,
            "finished": finished,
            "shots": shots,
            "metadata": {"example": True},
            "queued_position": random.randint(0, 50)
        })

    # add a stable demo job you can always show in presentation
    jobs.append({
        "job_id": "demo-0001",
        "backend_name": "ibmq_demo_backend",
        "status": "COMPLETED",
        "created": now - timedelta(minutes=5),
        "finished": now - timedelta(minutes=1),
        "shots": 1024,
        "metadata": {"demo":"true", "notes": "Guaranteed demo job"},
        "queued_position": 0
    })

    # sort by created desc
    jobs.sort(key=lambda x: x["created"], reverse=True)
    return jobs

# ---------- IBM CLIENT ----------
class IBMClient:
    def __init__(self, token=None, channel="ibm_quantum", verify_ibm_availability=True):
        """
        If token is None -> operate in mock mode.
        If token provided -> try to connect using QiskitRuntimeService.
        """
        self.token = token
        self.service = None
        self.mock = token is None
        if not self.mock and IBM_AVAILABLE and verify_ibm_availability:
            try:
                self.service = QiskitRuntimeService(channel=channel, token=self.token)
                self.mock = False
            except Exception as e:
                print("Failed to initialize IBM service:", e)
                self.service = None
                self.mock = True
        else:
            self.mock = True

    def list_jobs(self, limit=50):
        """
        Returns a list of job dicts with standardized fields:
        job_id, backend_name, status, created (datetime), finished (datetime or None), shots, metadata, queued_position
        """
        if self.mock:
            return _mock_jobs(min(limit, 50))
        else:
            jobs = []
            try:
                qjobs = self.service.jobs(limit=limit)
                for j in qjobs:
                    jdict = {
                        "job_id": j.job_id(),
                        "backend_name": j.backend().name() if hasattr(j.backend(), "name") else str(j.backend()),
                        "status": str(j.status()),
                        "created": getattr(j, "creation_date", None) or getattr(j, "time_created", None) or None,
                        "finished": getattr(j, "completion_date", None) or None,
                        "shots": None,
                        "metadata": getattr(j, "metadata", None),
                        "queued_position": getattr(j, "queue_position", None)
                    }
                    jobs.append(jdict)
            except Exception as e:
                print("Error fetching jobs from IBM:", e)
                return []
            return jobs

    def get_job(self, job_id):
        if self.mock:
            for j in _mock_jobs(50):
                if j["job_id"] == job_id:
                    return j
            return None
        else:
            try:
                job = self.service.job(job_id)
                jdict = {
                    "job_id": job.job_id(),
                    "backend_name": job.backend().name() if hasattr(job.backend(), "name") else str(job.backend()),
                    "status": str(job.status()),
                    "created": getattr(job, "creation_date", None) or getattr(job, "time_created", None) or None,
                    "finished": getattr(job, "completion_date", None) or None,
                    "shots": None,
                    "metadata": getattr(job, "metadata", None),
                    "queued_position": getattr(job, "queue_position", None)
                }
                return jdict
            except Exception as e:
                print("Error fetching job details:", e)
                return None
