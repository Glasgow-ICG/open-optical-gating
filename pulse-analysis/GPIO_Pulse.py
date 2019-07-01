# A scipt that takes delay, duration and GPIO pin as input 
# and outputs a 3.3V signal to the GPIO pin for the specified 
# duration after the specified delay

import sys
import RPi.GPIO as GPIO
import time

def pulse(delay,duration,pin):


	# Pulses for duration after delay
	time.sleep(delay)
	GPIO.output(pin,1)
	time.sleep(duration)
	GPIO.output(pin,0)
	


def init(pin):

	# Sets up board and pin
	GPIO.setmode(GPIO.BOARD)
	GPIO.setup(pin,GPIO.OUT)

	return 0

def clean():

	#Cleans up board
	GPIO.cleanup()
