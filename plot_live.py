import pandas as pd
import matplotlib.pyplot as plt
import time
from matplotlib.animation import FuncAnimation


# Initialize plot
fig, ax = plt.subplots()
line, = ax.plot([], [], label='Channel 1')
plt.legend()

def update(x_data,y_data): #update the plot with the new data it got
    try:
        # ax.set_xlim(x_data.min()-5 , x_data.max() + 5)
        # ax.set_ylim(y_data.min()-100 , y_data.min() + 100)
        line.set_data(x_data, y_data)
    except Exception as e:
        print(f"Error in animate function: {e}")

    return line

def plot_data(x_data,y_data):
    plt.style.use('fivethirtyeight')

    # Create animation
    ani = FuncAnimation(fig, update(x_data, y_data), interval=1000, cache_frame_data=False)

    plt.tight_layout()

    # Run the animation
    plt.show(block=True)

    # Keep the plot open (especially important in scripts)
    while plt.fignum_exists(fig.number):
        time.sleep(0.1)

#hello


