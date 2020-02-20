# Python imports
import fastpins as fp
import time
import timebox
import json


if __name__ == '__main__':

	delay = 500e3  # us
	
	data_file = open("settings.json")
	dict_data = json.load(data_file)
	
	laser_trigger_pin = dict_data['laser_trigger_pin']
	fluorescence_camera_pins = dict_data['fluorescence_camera_pins'] # Trigger, SYNC-A, SYNC-B
	
	duration = dict_data['fluorescence_exposure']  # us, this is the duration of a laser pulse (also controls the camera exposure if set to do so)


	if timebox.init_controls(laser_trigger_pin,fluorescence_camera_pins) == 0:

		# Tests laser and camera
		for i in range(10000):

			timebox.trigger_fluorescence_image_capture(delay, laser_trigger_pin, fluorescence_camera_pins, edge_trigger=False, duration=duration)
	else:
		print('Could not initialise laser and fluorescence camera')
