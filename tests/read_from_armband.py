import matplotlib.pyplot as plt
import numpy as np
from mindrove.board_shim import BoardShim
import time
from tests import armband_functions as funcs

start_time = time.time()
elapsed_time = time.time()-start_time
stop_time = 20

board_shim = funcs.connection() # connecting to the armband

emg_channels = BoardShim.get_emg_channels(board_shim.board_id)
sampling_rate = BoardShim.get_sampling_rate(board_shim.board_id)

window_size = 1 # seconds
num_points = window_size * sampling_rate
chanels_num = 8

emg_data = [[] for _ in range(chanels_num)]
unfiltered_emg_data = [[] for _ in range(chanels_num)]

samples = np.array([])


plt.figure()

print(board_shim.is_prepared())
while (elapsed_time <= stop_time+1):
  elapsed_time = time.time() - start_time

  if board_shim.get_board_data_count() >= num_points:
    funcs.get_new_data(unfiltered_emg_data,emg_data,emg_channels,num_points,board_shim,sampling_rate)

    for j in range(chanels_num):
      plt.subplot(4, 2, j + 1)
      samples = np.linspace(1, len(emg_data[0]), len(emg_data[0]))
      plt.plot(samples, emg_data[j])
      plt.xlim(0, len(samples))

    plt.tight_layout()
    plt.show()
