![kaijucapital](https://github.com/user-attachments/assets/aa312aea-d863-4690-9040-1bb89242f584)


## Quantitative Trading Strategy Development Framework

A robust, four-step framework for developing and validating quantitative trading strategies, designed to minimize overfitting and ensure statistical significance. This repository provides code and methodologies for backtesting, permutation testing, and walk-forward analysis, illustrated with a moving average crossover strategy.

---

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [Data Preparation](#1-data-preparation)
  - [In-Sample Optimization](#2-in-sample-optimization)
  - [Permutation Testing](#3-in-sample-permutation-test)
  - [Walk Forward Analysis](#4-walk-forward-test)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Disclaimer](#disclaimer)

---

## Features
- **In-Sample Excellence**: Parameter optimization for strategies (e.g., moving average crossovers).
- **Permutation Tests**: Statistical validation to rule out random luck.
- **Walk Forward Testing**: Out-of-sample validation with rolling windows.
- **Risk Management**: Includes transaction costs, position sizing, and stop-loss examples.
- **Visualization**: Equity curve plotting and performance metrics.

---

## Installation

### Prerequisites
Ensure you have Python 3.8 or higher installed on your system. You can verify this by running:
```bash
python --version
```

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/trading-strategy-framework.git
   cd trading-strategy-framework
   ```
2. Install dependencies:
   ```bash
   pip install pandas numpy yfinance matplotlib scipy
   ```

---

## Usage

### 1. Data Preparation
Fetch historical price data using Yahoo Finance:
```python
import yfinance as yf
data = yf.download("AAPL", start="2018-01-01", end="2023-01-01")
```

### 2. In-Sample Optimization
Optimize strategy parameters on historical data:
```python
from strategy import optimize_ma_parameters
best_params, sharpe = optimize_ma_parameters(data)
print(f"Optimal Parameters: {best_params}, Sharpe: {sharpe:.2f}")
```

### 3. In-Sample Permutation Test
Validate statistical significance:
```python
from permutation import in_sample_permutation_test
p_value = in_sample_permutation_test(data, best_params)
print(f"P-value: {p_value:.4f}")  # Significant if < 0.05
```

### 4. Walk Forward Test
Evaluate out-of-sample performance:
```python
from walk_forward import run_walk_forward_test
out_of_sample_returns = run_walk_forward_test(data, train_years=3, test_years=1)
```

### 5. Walk Forward Permutation Test
Validate out-of-sample robustness:
```python
from permutation import walk_forward_permutation_test
p_value_out = walk_forward_permutation_test(out_of_sample_returns)
print(f"Out-of-Sample P-value: {p_value_out:.4f}")
```

### Example Script
Run the full pipeline:
```bash
python main.py --ticker AAPL --start 2018-01-01 --end 2023-01-01
```

---

## Results
- **In-Sample Sharpe Ratio**: 1.25 (p = 0.03)
- **Walk Forward Sharpe Ratio**: 0.95 (p = 0.04)
- **Equity Curve**:  
  ![Equity Curve](equity_curve.png)

---

## Contributing
We welcome contributions! To contribute:
1. Fork the repository.
2. Create a feature branch:
   ```bash
   git checkout -b feature/your-feature
   ```
3. Commit your changes:
   ```bash
   git commit -m 'Add some feature'
   ```
4. Push to the branch:
   ```bash
   git push origin feature/your-feature
   ```
5. Open a Pull Request and describe your changes.

---

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Disclaimer
This code is for educational and research purposes only. **Do not use it for live trading without further validation.** Past performance does not guarantee future results. The authors are not responsible for any financial losses.

---


