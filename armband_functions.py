from mindrove.board_shim import BoardShim, MindRoveInputParams, BoardIds, MindroveConfigMode
from mindrove.data_filter import DataFilter, FilterTypes, DetrendOperations, NoiseTypes

import numpy as np

adjustment_to_microvolts = 0.045


def connection(): # connect to the armband board
    BoardShim.enable_dev_board_logger()  # enable logger when developing to catch relevant logs
    params = MindRoveInputParams()
    board_id = BoardIds.MINDROVE_WIFI_BOARD
    board_shim = BoardShim(board_id, params)

    board_shim.prepare_session()
    board_shim.start_stream()
    board_shim.config_board(MindroveConfigMode.EEG_MODE)  # switch to eeg mode

    return board_shim

def filters(data, sampling_rate): #filtering the data

    #DataFilter.perform_highpass(data=data, sampling_rate=sampling_rate, cutoff=20, order=2, filter_type=FilterTypes.BUTTERWORTH, ripple=0)
    #DataFilter.perform_lowpass(data=data, sampling_rate=sampling_rate, cutoff=500, order=2, filter_type=FilterTypes.BUTTERWORTH, ripple=0)
    DataFilter.remove_environmental_noise(data, sampling_rate, NoiseTypes.FIFTY)
    DataFilter.perform_bandpass(data=data, sampling_rate=sampling_rate, center_freq = 125, band_width=sampling_rate/2, order=1, filter_type=FilterTypes.BUTTERWORTH, ripple=0)


def get_new_data(unfiltered_previous_data,previous_data,data_channels , num_points , board_shim, sampling_rate ):
    temp = np.array([])
    new_data = board_shim.get_board_data(num_points)[data_channels] #getting the new data samples from the armband
    for i in range (len(data_channels)): #filters and enter the new data
        unfiltered_previous_data[i] = np.append(unfiltered_previous_data[i], new_data[i] * adjustment_to_microvolts)
        temp = unfiltered_previous_data[i]
        filters(unfiltered_previous_data[i],sampling_rate)
        previous_data[i] = unfiltered_previous_data[i]
        unfiltered_previous_data[i] = temp

    return  unfiltered_previous_data,new_data





























