
# MSBD6000M Assignment 1

This repository contains code and a report for a discrete‐time asset‐allocation problem solved via Q‐learning under CARA (negative exponential) utility.

## Authors and Contributions

**WONG Chong Ki (20978851)**  
MSc in Big Data Technology, HKUST  
Email: ckwongch@connect.ust.hk  

**WENG Yanbing (21091234)**  
MSc in Big Data Technology, HKUST  
Email: ywengae@connect.ust.hk  

Both authors contributed equally to the codebase and the report, each responsible for 50% of the overall work.

## Files Overview

- **MSBD6000M_Assignment1.ipynb**  
  Jupyter Notebook containing all experiment code cells. Run this notebook to reproduce training and visualizations for multiple scenarios.

- **msbd6000m_assignment1.py**  
  Python script version of the Q‐learning code. It includes:
  - `Environment` class: sets up the Markov Decision Process (state transitions, utility, etc.).
  - `Agent` class: Q‐learning logic with \(\epsilon\)-greedy exploration.
  - `Trainer` class: orchestrates the learning process, logs errors and final wealth, and plots results.

- **MSBD6000M_Assignment_1_report.pdf**  
  Overleaf‐generated PDF report. This includes:
  - Analytical derivation under CARA utility,
  - Detailed explanation of the code’s approach,
  - Experiment results with four scenarios.

- **uni_test.py**  
  A Python unittest script that verifies correctness of the environment, agent, and trainer. It checks transitions, utility calculations, Q-table updates, and so on. Run it via:
  ```bash
  python uni_test.py

---

For more details, please refer to the **report PDF**.
