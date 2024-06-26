# -*- coding: utf-8 -*-
"""solution.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Ft_Ax0SUofOK5BGHKMnHNAhz9Yoih2pM

# 0. The Obligatory Part
"""

!pip3 install gurobipy

import math
import pandas as pd
import gurobipy as gp
from gurobipy import GRB

"""# 1. Define the Data Structures"""

path = '/content/drive/MyDrive/Tadika Mesra Bunga Matahari/#1 Optimization Problem/challenge2_stock-selection/data/etf_data_fix.xlsx'

"""### Pre-Processing: Raw Data"""

stocks_df = pd.read_excel(path)

stocks_df

# Extract necessary data
stock_names = stocks_df['ticker'].tolist()
stock_aar = stocks_df['annualized_avg_return_dec'].tolist()
stock_risk = stocks_df['5y_risk'].tolist()

"""<br> <br>



---




# 2. Construct the Model

We have a sets $\mathcal{I}$ of $\texttt{stocks}$. Then we have parameters and scalars. So let:


$$
\begin{align*}
i & \quad \text{stock} \\
n & \quad \text{initial investment} \\
r & \quad \text{individual stock risk} \\
t & \quad \text{individual stock return} \\
a & \quad \text{annualized average return} \\
\end{align*}
\\
$$

and the following decision variables:

$$
\begin{align*}
x_i & \quad \text{weight of investment allocation} \\
y_i & \quad \text{stock selection decision} \\
\end{align*}
\\ \\
$$

The objective is we want to find optimal investment allocation that maximize the annualized average return. That subject to the constraints:
1. Portfolio risk below than 15%
2. Total portfolio return more than 20%
3. Minimum investment allocation is distributed to 3 stocks
4. Total of $x_i$ is 100% and $0 \leq x_i \leq 100$

### **Objective Function**

$$
\begin{align*}
{\rm maximize} & \quad \displaystyle \sum_{i\in \mathcal{I}} \ a_i \cdot x_i & \quad \forall i\in {\mathcal{I}} \\ \\
{\rm s.t.} & \quad \displaystyle \sqrt{\ \sum_{i \in \mathcal{I}} \ r_i^2 \cdot x_i^2} < 15\% & \quad \forall i \in \mathcal{I}
\\
& \quad \displaystyle \sum_{i \in \mathcal{I}} a_i \cdot x_i > 20 \% & \quad \forall i\in {\mathcal{I}}
\\
& \quad \displaystyle \sum_{i \in \mathcal{I}} y_i \geq 3 & \quad \forall i\in {\mathcal{I}}
\\
& \quad \displaystyle \sum_{i \in \mathcal{I}} x_i = 100 \% & \quad \forall i\in {\mathcal{I}}
\\
& \quad x_i \in \{0...100\} &  \quad \forall i\in {\mathcal{I}}
\\
& \quad x_i \geq 0 \ ; \ x_i \leq 100
\\ \\
\end{align*}
$$
"""

# Initialize Gurobi model
model = gp.Model("PortfolioOptimization")

"""# 3. Build the Decision Variables"""

# Define variables
allocation_vars = model.addVars(stock_names, name="allocation", lb=0.0, ub=1.0)
stock_selection_vars = model.addVars(stock_names, name="select", vtype=GRB.BINARY)

"""
# 4. Subject to the Constraints


for the first constraints, that is

$$
\begin{align*}
\displaystyle \sqrt{\ \sum_{i \in \mathcal{I}} \ r_i^2 \cdot x_i^2} < 15\% & \quad \forall i \in \mathcal{I}
\\
\end{align*}
$$

Ensure portoflio risk is less than 15%"""

# constraint 1: portfolio risk <= 15%
risk_expr = gp.quicksum((stock_risk[i] ** 2) * (allocation_vars[stock] ** 2) for i, stock in enumerate(stock_names))
model.addQConstr(risk_expr <= 0.15 ** 2, "RiskLimit")

"""Then for the second constraint, that is

$$
\begin{align*}
& \quad \displaystyle \sum_{i \in \mathcal{I}} a_i \cdot x_i > 20 \% & \quad \forall i\in {\mathcal{I}}
\\
\end{align*}
$$

Ensure total portfolio return is greater than 20%
"""

# constraint 2
model.addConstr(gp.quicksum(allocation_vars[stock] * stock_aar[i] for i, stock in enumerate(stock_names)) >= 0.2, "MaxReturn")

"""For the third constraint, that is

$$
\begin{align*}
& \quad \displaystyle \sum_{i \in \mathcal{I}} y_i \geq 3 & \quad \forall i\in {\mathcal{I}}
\\
& \quad \displaystyle x_i \geq 5.25% & \quad \forall i\in {\mathcal{I}}
\\
\end{align*}
$$

Ensure selected stocks is more than or equal to 3 stock with minimum allocation is 5.25%
"""

# constraint 3
model.addConstr(gp.quicksum(stock_selection_vars[stock] for stock in stock_names) >= 3)

# Gurobi constraints for setting lower bound of allocation if stock is selected
for stock in stock_names:
  model.addConstr(allocation_vars[stock] >= 0.0525 * stock_selection_vars[stock])

"""For the fourth constraint, that is

$$
\begin{align*}
& \quad \displaystyle \sum_{i \in \mathcal{I}} x_i = 100 \% & \quad \forall i\in {\mathcal{I}}
\\
\end{align*}
$$

Ensure total portfolio return is greater than 20%
"""

# constraint 4
# Budget constraint: sum of allocations must not exceed 100% of the portfolio
model.addConstr(allocation_vars.sum() <= 1, "Budget")

"""# 5. Set The Objective Function

### **Objective Function**

$$
\begin{align*}
{\rm maximize} & \quad \displaystyle \sum_{i\in \mathcal{I}} \ a_i \cdot x_i & \quad \forall i\in {\mathcal{I}} \\ \\
\end{align*}
$$
"""

# Set objective: Maximize total expected return
model.setObjective(gp.quicksum(allocation_vars[stock] * stock_aar[i] for i, stock in enumerate(stock_names)), GRB.MAXIMIZE)

"""# 6. Solve the Model"""

# solve the model
model.optimize()

# Print solution
if model.status == GRB.OPTIMAL:
  print("\n     OPTIMAL solution found!     \n")
  total_expected_annual_return = 0  # Initialize total expected return
  portfolio_risk_squared = 0  # Initialize the squared risk sum for the portfolio

  for i, stock in enumerate(stock_names):
    if allocation_vars[stock].X > 0:
      allocation_percentage = allocation_vars[stock].X * 100 # convert to percentage
      expected_return_contribution = allocation_vars[stock].X * stock_aar[i] * 100 #convert to percentage
      total_expected_annual_return += expected_return_contribution

      # Update the squared risk sum
      portfolio_risk_squared += (stock_risk[i] ** 2) * (allocation_vars[stock].X ** 2)

      print(f"{stock}\t allocation: {allocation_percentage:.2f}%\tExpected annual return contribution: ${expected_return_contribution:.2f}%")

  # Calculate the portfolio risk as the square root of the accumulated squared risks
  portfolio_risk = math.sqrt(portfolio_risk_squared) * 100 # Convert to percentage

  print('\n')
  print(f"Total expected annual return: {total_expected_annual_return:.2f}%")
  print(f"Portfolio risk: {portfolio_risk:.2f}%")

else:
    print("No feasible solution found.")