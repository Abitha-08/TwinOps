TwinOps Starter - Modules 1 to 5
================================

IMPORTANT
---------
The supplied ZIP contained 14 PNG screen designs, not application source code.
This starter app recreates the first five selected screens as a working demo.
It uses generated sample metrics. It does NOT monitor a real website yet.
The Digital Twin and simulation are simplified illustrative models.

REQUIREMENTS
------------
- Windows 10/11
- Python 3.10 or newer
- Internet connection for the first package installation

HOW TO RUN (Windows)
--------------------
1. Extract TwinOps_Starter.zip.
2. Open the extracted TwinOps_Starter folder.
3. Click the folder address bar, type cmd, and press Enter.
4. Create a virtual environment:
      py -m venv .venv
5. Activate it:
      .venv\Scripts\activate
6. Install packages:
      python -m pip install --upgrade pip
      pip install -r requirements.txt
7. Start the application:
      python -m streamlit run app.py
8. Browser should open automatically. If not, open:
      http://localhost:8501

DEMO LOGIN
----------
Email: demo@twinops.com
Password: twinops123

WHAT TO CLICK
-------------
- Dashboard: overview cards and performance trend chart.
- Digital Twin Topology: conceptual Users -> Web -> Service -> Database view.
- ML & SLM Analysis:
    * Anomaly Detection: choose Normal, High CPU, High Memory or High Error scenario.
    * Prediction: see Random Forest next-sample CPU estimate and test MAE.
    * SLM Analysis: click Generate SLM analysis. If Ollama is unavailable,
      the app displays a labelled fallback explanation.
- Simulation & Trust: change proposed connection-pool size and view illustrative
  before/after estimates. Approval is for human review only; no live changes occur.
- Logout: return to login screen.

OPTIONAL REAL SLM WITH OLLAMA
-----------------------------
1. Install Ollama from https://ollama.com/download
2. Open a new terminal and run:
      ollama pull llama3.2
3. Ensure Ollama is running, then use the SLM Analysis tab and model name llama3.2.
The app calls Ollama at http://localhost:11434.

TROUBLESHOOTING
---------------
- If 'py' is not recognized, install Python from https://www.python.org/downloads/
  and select 'Add Python to PATH' during installation. Then reopen CMD.
- If port 8501 is busy:
      python -m streamlit run app.py --server.port 8502
  Then open http://localhost:8502
- To stop the app, click the terminal and press Ctrl+C.

NEXT STEP FOR A REAL PROJECT
----------------------------
Replace generated sample metrics with metrics collected from a controlled test
application (for example, your own Spring Boot app). Validate ML models on real
historical data, build a calibrated simulation, and require human approval before
any real configuration changes.
