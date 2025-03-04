### 3. Walk Forward Test
   - **Purpose**: This step evaluates how the strategy performs on out-of-sample data to assess its robustness and real-world applicability. It simulates trading in real-time by moving through time incrementally.
   - **Process**:
     - Divide the historical data into rolling windows: an in-sample period (for optimization) and an out-of-sample period (for testing).
     - For each window:
       - Optimize the strategy parameters on the in-sample data (e.g., the first 3 years of data).
       - Apply the optimized parameters to the subsequent out-of-sample data (e.g., the next 1 year) without further adjustments.
       - Move the window forward (e.g., by 1 year) and repeat the process until the entire dataset is covered.
     - Measure the strategy's performance (e.g., returns, drawdowns, win rate) across all out-of-sample periods.
   - **Outcome**: A successful walk forward test shows that the strategy can adapt to new data and maintain performance, reducing the risk of overfitting to the initial in-sample data.
