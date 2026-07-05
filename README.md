# Quantitative Modeling of Market Inefficiencies
## A Behavioral Economics Approach to Sports Betting Markets

This repository contains a quantitative financial analytics framework designed to identify structural inefficiencies and behavioral anomalies within sports betting markets. By leveraging predictive statistical modeling and contrasting it with public market odds, the system isolates emotional overreactions and market mispricings, applying corporate risk management principles to optimize capital growth.

---

### 📊 Project Architecture & Methodology

The core framework bridges predictive data science with mathematical finance through two main layers:

1. **The Statistical Predictive Engine (Poisson Distribution)**
   - Utilizes independent Poisson probability mass functions (PMF) parameterized by Team Expected Goals ($xG$) to compute the probability of exact score lines.
   - Computes a joint probability distribution matrix to yield precise mathematical probabilities for macro-market outcomes: Home Win ($P_H$), Draw ($P_D$), and Away Win ($P_A$).

2. **The Capital Allocation Layer (Kelly Criterion)**
   - To counteract market variance and safeguard institutional bankrolls, the system incorporates the **Kelly Criterion** for optimal sequential asset allocation:
     $$f^* = \frac{bp - q}{b}$$
     *Where $b$ represents the net market odds ($\text{Decimal Odds} - 1$), $p$ is the algorithmic probability of success, and $q$ is the probability of failure ($1 - p$).*
   - Execution filters out negative expected value ($\mathbb{E}[X] \le 0$) scenarios, mitigating capital exposure when market prices exhibit strict efficiency.

---

### 📈 Empirical Validation & Backtesting

The model's underlying code has been rigorously cross-validated using historical data environments:
* **Data Environment:** Validated against multi-season football performance records (`E0.csv` data streams).
* **Predictive Accuracy:** Empirical backtesting over full historical fixtures demonstrated a **46.32% overall forecasting accuracy**, successfully validating the statistical baseline against public market closing lines.

---

### 🖥️ Technology Stack

* **Core Language:** Python 3
* **Scientific Computing:** `NumPy` (matrix outer products), `SciPy` (Poisson statistics orchestration)
* **Frontend UI Framework:** `Streamlit` (Reactive dashboards for real-time probability and capital allocation mapping)

---

### 🚀 Local Execution

To run the interactive quantitative dashboard locally, configure your workspace environment and execute the Streamlit application:

```bash
# Install dependencies
pip install -r requirements.txt

# Launch the interactive model
py -3 -m streamlit run streamlit_app.py
