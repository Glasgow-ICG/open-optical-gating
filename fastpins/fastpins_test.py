# A python script to test the successful importing of the fastpins module
import fastpins as fp

# Default set that sets up some pins for reading and reads their value
def simple_read_test():

	# Pins to be read
	pins = (10,12)

	# Initialises board
	fp.init()

	# Sets both pins for reading
	fp.setpin(pins[0],0,0) # no pud
	fp.setpin(pins[1],0,2) # pull down resitor

	# Reads in pin value
	for i in range(len(pins)):
		print(str(pins[i])+': '+str(fp.read(pins[i])))

# A pulse test that pulses pin 8 on and off with an interval of 10us
def pulse_test():
	
	# Defines pin to be used
	pin = 8

	# Initialises board and pin
	fp.init()
	fp.setpin(pin,1,0)

	# Pulses the pin
	for i in range(100000):
		fp.pulse(100,100,pin)

	
simple_read_test()
pulse_test()
