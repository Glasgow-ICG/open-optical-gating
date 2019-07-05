# A python script to test the Mecury movement stages
import serial


# Function to test addresses
def test_address():

	# Opens the usb port and sets initial parameters
	encoding = 'utf-8'
	timeout = 3
	request_info_command = 'CS'
	bites_max = 100
	ser = serial.Serial('/dev/ttyUSB0',timeout=timeout)

	# Prints inforamtion to user
	print('USB Information:\n'+str(ser))
	print('Command sent: '+request_info_command)
	print('Encoding: '+encoding)
	print('Maximum bites read: '+str(bites_max)+'\n')

#	ser.write(('SVO A 1\nMOV A 0\n').encode(encoding))
	# Loops through all addresses, selecting them, sending a command and then waits for a response.
	for i in range(48,71):
		
		# As the ASCII 58-64 are not used in numbering
		if i < 58 or i > 64: 

			# Performs the following: selects address, sends command, listens for response. 			
			ser.write((chr(1)+chr(i)+','+request_info_command+'\r').encode(encoding))


			# Checks response
			if ser.read(bites_max):
				# Correct naming convention
				if i < 58:
					print('Address '+str(i-48)+' has responded!')
				else:
					print('Address '+str(i-55)+' has responded!')

			else:

				# Correct naming convention
				if i < 58:
					print('Address '+str(i-48)+' timed out.')
				else:
					print('Address '+str(i-55)+' timed out.')
	# Closes the USB port
	ser.close()
	print('\nUSB port closed.')


test_address()
