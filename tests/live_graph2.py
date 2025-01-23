import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time

plt.style.use('fivethirtyeight')

# Initialize plot
fig, ax = plt.subplots()
line1, = ax.plot([], [], label='Channel 1')
line2, = ax.plot([], [], label='Channel 2')
plt.legend()


def animate(i):
    try:
        # Read new data for each animation frame
        data = pd.read_csv('../../MindRoveSDK/originals/data.csv')

        # Debug: Print out the data being read
        print(f"Frame {i}: Read data:\n", data.head())

        x = data['x_value']
        y1 = data['total_1']
        y2 = data['total_2']

        # Update line data
        line1.set_data(x, y1)
        line2.set_data(x, y2)

        # Update axis limits if needed
        ax.set_xlim(x.min(), x.max() + 5)
        current_ymax = max(y1.max(), y2.max())
        current_ymin = min(y1.min(), y2.min())
        ax.set_ylim(current_ymin - 5, current_ymax + 5)

    except Exception as e:
        print(f"Error in animate function: {e}")

    return line1, line2


# Create animation
ani = FuncAnimation(fig, animate, interval=1000, cache_frame_data=False)

plt.tight_layout()

# Run the animation
plt.show(block=True)

# Keep the plot open (especially important in scripts)
while plt.fignum_exists(fig.number):
    time.sleep(0.1)
