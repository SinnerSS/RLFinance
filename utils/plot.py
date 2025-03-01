import matplotlib.pyplot as plt

def plot_values(history):
    """Plot portfolio value"""

    fig, ax = plt.subplots(figsize=(12, 8))

    ax.plot(history['date'], history['portfolio_value'])
    ax.set_title('Portfolio Value Over Time')
    ax.set_ylabel('Value ($)')
    ax.grid(True)

    plt.tight_layout()
    plt.show()

    return fig
