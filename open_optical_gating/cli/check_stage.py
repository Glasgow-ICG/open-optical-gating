# A python script to test the Mercury stages
import serial


# Function to test addresses
def test_address():

    # Opens the usb port and sets initial parameters
    timeout = 0.1
    baud_rate = 57600
    bytesize = 8
    parity = "N"
    stopbits = 1
    xonxoff = True
    ser = serial.Serial(
        "/dev/ttyUSB0",
        timeout=timeout,
        baud_rate=baud_rate,
        bytesize=bytesize,
        parity=parity,
        stopbits=stopbits,
        xonxoff=xonxoff,
    )

    encoding = "utf-8"
    request_info_command = "ID"
    terminate = chr(13) + chr(10)
    command_value = "?"

    # Prints inforamtion to user
    print("USB Information:\n" + str(ser))
    print("Command sent: " + request_info_command)
    print("Encoding: " + encoding + "\n")

    # Loops through all addresses, selecting them, sending a command and then waits for a response.
    for i in range(1, 32):

        # Sends command
        command = str(i) + request_info_command + command_value + terminate
        ser.write(command.encode(encoding))

        # Gets and prints recieved signal
        read_signal = ser.readline()
        print("Address " + str(i) + ":\t" + read_signal.decode(encoding))

    return ser


# Creates a terminal command for the stages
def interactive_commands(ser):

    # Informs user
    print("Entering interactive Newport shell.\nEnter q to quit.")

    # Sets command values
    encoding = "utf-8"
    terminate = chr(13) + chr(10)

    # Creates an inf loop until user exits
    while True:

        # Gets user command
        command = input("Command:\n")

        # Checks if command is the exit command
        if command == "q":
            break

        else:
            # Sends the command
            ser.write((command + terminate).encode(encoding))

            # Reads result and prints output to screen
            ser_read = ser.readline()
            print(ser_read.decode(encoding))

    # Closes USB port
    ser.close()
    print("USB port closed.")


if __name__ == "__main__":
    # Tests address and runs user command prompt
    ser = test_address()
    interactive_commands(ser)
