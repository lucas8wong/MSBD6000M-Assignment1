```markdown
# MSBD6000M Assignment 1

This repository contains code and a report for a discrete‐time asset‐allocation problem solved via Q‐learning under CARA (negative exponential) utility.

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
  python -m unittest uni_test.py
  ```
  to ensure everything passes.

- **README.md**  
  This file. Summarizes the purpose and contents of the repository.

## How to Run

1. **Option 1: Python script**  
   - Ensure you have Python 3, NumPy, Matplotlib installed.  
   - Run `python msbd6000m_assignment1.py`.  
   - The script will train an agent for multiple scenarios and display the convergence plots.

2. **Option 2: Jupyter Notebook**  
   - Open `MSBD6000M_Assignment1.ipynb` in Jupyter or any compatible environment.  
   - Execute the cells sequentially to reproduce training logs and plots.

3. **Option 3: Unit Tests**  
   - Run the tests via:
     ```bash
     python -m unittest uni_test.py
     ```
   - This checks environment transitions, utility correctness, Q-table updates, etc., ensuring code integrity.

## Key Highlights

- **CARA Utility**: Negative‐exponential utility ensures a closed‐form solution exists, which the Q‐learning agent also approximates numerically.
- **Multiple Market Scenarios**: Each with different \(\{p, a_{\text{ret}}, b_{\text{ret}}, r\}\) to assess policy convergence under varying risk/return conditions.
- **Results**: In favorable (high‐probability/high‐return) cases, final wealth tends to saturate at or near the upper bound. In moderate scenarios, the agent still learns a policy that leverages net positive expectation.

## Authors and Contributions

**WONG Chong Ki (20978851)**  
MSc in Big Data Technology, HKUST  
Email: ckwongch@connect.ust.hk  

**WENG Yanbing (21091234)**  
MSc in Big Data Technology, HKUST  
Email: ywengae@connect.ust.hk  

Both authors contributed equally to the codebase and the report, each responsible for 50% of the overall work.

---

For more details, please refer to the **report PDF**.
```
