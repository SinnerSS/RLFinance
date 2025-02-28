import matplotlib.pyplot as plt
from typing import Dict

import matplotlib.pyplot as plt
from typing import Dict

def plot_values(history: Dict):
    """Plot portfolio value"""

    dates = history.keys()
    values = history.values()

    fig, ax = plt.subplots(figsize=(12, 8))

    ax.plot(dates, values)
    ax.set_title('Portfolio Value Over Time')
    ax.set_ylabel('Value ($)')
    ax.grid(True)

    plt.tight_layout()
    plt.show()
