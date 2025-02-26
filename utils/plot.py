import matplotlib.pyplot as plt
from typing import Dict

import matplotlib.pyplot as plt
from typing import Dict

def plot_performance(performance: Dict):
    """Plot portfolio value"""
    portfolio_values = performance['values']

    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot portfolio value
    ax.plot(portfolio_values['date'], portfolio_values['portfolio_value'])
    ax.set_title('Portfolio Value Over Time')
    ax.set_ylabel('Value ($)')
    ax.grid(True)

    plt.tight_layout()
    plt.show()
