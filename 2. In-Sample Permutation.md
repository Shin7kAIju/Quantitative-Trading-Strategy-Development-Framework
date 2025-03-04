### 2. In-Sample Permutation Test
   - **Purpose**: This step tests whether the strategy's performance in the in-sample data is statistically significant or if it could have occurred by random chance.
   - **Process**:
     - Use a permutation test (also called a randomization test) to shuffle or randomize the in-sample data (e.g., price returns or other inputs) while keeping the strategy rules intact.
     - Run the strategy on these randomized datasets multiple times (e.g., 1,000 permutations) and compare the original strategy's performance to the distribution of results from the randomized data.
     - If the strategy's performance (e.g., returns, Sharpe ratio) is significantly better than the randomized results (e.g., above the 95th percentile), it suggests the strategy has genuine predictive power, not just luck.
   - **Outcome**: This helps validate that the strategy's success in the in-sample data is not due to random noise or overfitting.
