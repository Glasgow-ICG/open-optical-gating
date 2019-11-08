# Python imports
import re
import time

# A function that ensures the stage is set to the correct mode
def set_stage_to_recieve_input(ser,address,encoding,terminator):

	# Function inputs:
	#	ser = the serial object for the usb (already set up in initialisation function)
	#	address = the address number (int) of the stage to ensure is in the correct state
	#	encoding = the encoding needed for the stage controllers (usually utf-8)
	#	terminator = the terminating characters sent at the end of each command


	# To-do:
	#	- Perform error checks rather than printing responses
	#	- Set up stage parameters (increment, acceleration, velocity etc)
	#	- Read current state rather than waiting for a long time

	# Information:
	#	The easiest way to do this is to reset each controller on start up so the initial state is known.
	#	Currently, all initial parameters are left but if they need to be changed should be done so in this function.

	#Command list
	reset = 'RS'
	ready = 'OR'
	config = 'PW'
	request_info = 'VE'
	set_neg_software_limit = 'SL'
	new_neg_software_limit = '0.01'

	# Checks address is correct
	command = str(address)+str(request_info)+str(terminator)
	ser.write(command.encode(encoding))
	response = ser.readline()
	if not response:
		print('No information recieved when requesting version information.\nCommand sent: '+command)
		return 1

	# Resets the controller
	command = str(address)+str(reset)+str(terminator)
	ser.write(command.encode(encoding))
	response = ser.readline()
	if response:
		print(response.decode(encoding))


	# Resetting the controller takes time
	time.sleep(10)

#	# Enters config stage (currently does nothing but here for easy of adding configuration"
#	command = str(address)+str(config)+str(0)+str(terminator)
#	ser.write(command.encode(encoding))
#	response = ser.readline()
#	if response:
#		print(response.decode(encoding))
#
#	# Leaves config state
#	command = str(address)+str(config)+str(1)+str(terminator)
#	ser.write(command.encode(encoding))
#	response = ser.readline()
#	if response:
#		print(response.decode(encoding))

	# Readies controller
	command = str(address)+str(ready)+str(terminator)
	ser.write(command.encode(encoding))
	response = ser.readline()
	if response:
		print(response.decode(encoding))

	# Decreases negative software limit to prevent being stuck in disabled state
	# Currently, whenever restarted the controller 'thinks' its at position 0 (regardless of actual position)
	# Thus, if immediately entered into disabled state I can not be re-entered into ready state as this is at the edge of the software limit
	# A (sort of) solution is to set the negative software limit from 0 to -0.1 or some other small number below 0
	command = str(address)+set_neg_software_limit+new_neg_software_limit+str(terminator)
	ser.write(command.encode(encoding))

	return 0

# Moves stage by an increment to move the zebrafish through the plane
def move_stage(ser, address, increment, encoding, terminate):


	# Function inputs
	#		ser = the serial object return by the init_controls function
	#		address = the address of the stage to be moved
	#		increment = the increment to move the stage by (float)
	#		encoding = the type of encoding to be used to send/recieve commands (usually 'utf-8') (str)
	#		terminate = the termination character set (str)

	# To-dos:
	#		- Function should check errors errors sent from stage
	#		- Function should check what mode controller is in and change to ready mode (if not in ready mode)
	#		- Function should return error in current position
	#		- Should read current state rather than waiting a long time

	# Gets the current stage position
	command = str(address) + 'TP?' + str(terminate)
	ser.write(command.encode(encoding))
	response = (ser.readline()).decode(encoding)

	# Tests if response present and if so extracts position information and tests move validity
	if not response:

		print('Error. No response obtained from stage '+str(address)+'.\nCheck stage is switched on and responsive.')

		return 1

	else:
		current_position = float(re.sub(str(address)+'TP','',response)) #Extracts position from response
#		print('Current position: '+str(current_position))

		# Gets negative software limit
		command = str(address)+'SL?'+str(terminate)
		ser.write(command.encode(encoding))
		response = (ser.readline()).decode(encoding)
		negative_software_limit = float(re.sub(str(address)+'SL','',response))
#		print(negative_software_limit)

		#Gets positive software limit
		command = str(address)+'SR?'+str(terminate)
		ser.write(command.encode(encoding))
		response = (ser.readline()).decode(encoding)
		positive_software_limit = float(re.sub(str(address)+'SR','',response))
#		print(positive_software_limit)

		# Checks if movement request is within software limit
		if (current_position + float(increment)) > negative_software_limit and (current_position + float(increment)) < positive_software_limit :

			# Moves the stage
			command = str(address)+'PR'+str(increment)+str(terminate)
			ser.write(command.encode(encoding))

#			# Gets the new position
#			command = str(address) + 'TP?' + str(terminate)
#			ser.write(command.encode(encoding))
#			response = (ser.readline()).decode(encoding)
#			current_position = float(re.sub(str(address)+'TP','',response)) #Extracts position from response
#			print('Current position after move: '+str(current_position))

			return 0

		else:

			print('Error. Cannot move stage by increment '+str(increment)+' at position '+str(current_position)+' as would be outside software limits of '+str(negative_software_limit) + ' - ' + str(positive_software_limit) )
			command = str(address)+'PA'+str(0)+str(terminate)
			ser.write(command.encode(encoding))
			return 2




# Gets the user set limits of the of the specific address
def set_user_stage_limits(ser, address, encoding, terminator):

	# Function inputs:
	#	ser = the serial object for the usb
	#	address = the address number (int or str) for the stage to be controlled
	#	terminator = the required string to terminate a command
	#	encoding = the encoding needed for the usb stages (str) (usually 'utf-8')


	# Commands
	move_absolute = 'PA'
	get_current_position = 'TP'
	toggle_disable_state = 'MM'

	# Activates the stages to recieve input and checks the result
	if set_stage_to_recieve_input(ser, address, encoding, terminator) != 0:
		print('Could not set user stage limits. Check stages.')
		return 1

	# Disables the stages and gets user to move to edge of the zebrafish heart
	command = str(address)+toggle_disable_state+str(0)+terminator
	ser.write(command.encode(encoding))

	# Waits the user to move the stage
	input('Please move the stage to the edge of the zebrafish heart.\nPress any key when the stage is in the correct positon.')

	# Gets the position of the stage
	command = str(address)+get_current_position+terminator
	ser.write(command.encode(encoding))
	response = (ser.readline()).decode(encoding)
	first_limit = float(re.sub(str(address)+get_current_position,'',response))
	print(first_limit)

	# Waits the user to move the stage
	input('Please move the stage to the other edge of the zebrafish heart.\nPress any key when the stage is in the correct positon.')

	# Gets the position of the stage
	command = str(address)+get_current_position+terminator
	ser.write(command.encode(encoding))
	response = (ser.readline()).decode(encoding)
	second_limit = float(re.sub(str(address)+get_current_position,'',response))
	print(second_limit)

	# Checks acquired position with user.
	print('Limits acquired. The stage will now move to from the one edge of the heart, to the other and then back again.\nPlease check to see if the limits are correct.')
	relative_increment = first_limit - second_limit
	print(relative_increment)
	time.sleep(3) # Gives the user a chance to read

		# Enables the stage
	command = str(address)+toggle_disable_state+str(1)+terminator
	ser.write(command.encode(encoding))

		# Moves the stages
	stage_response, current_position = move_stage(ser, address, relative_increment, encoding, terminator)

	if stage_response != 0:
		return None, None, None

	stage_response, current_position = move_stage(ser, address, -relative_increment, encoding, terminator)

	if stage_response != 0:
		return None, None, None

		# Checks results with user
	correct = input('Were these limits correct?\nPress y for yes or anything else for no.')
		# If results are incorrect, relaunches function
	if correct != 'y':
		first_limit, second_limit, current_position = set_user_stage_limits(ser, address, encoding, terminator)

	# Returns limits in the order of smallest, largest
	if second_limit > first_limit:

		# Ensures stage is at the smallest limit

		stage_response, current_position = move_stage(ser,address, relative_increment, encoding, terminator)
		return first_limit, second_limit, current_position
	else:
		return second_limit, first_limit, current_position
