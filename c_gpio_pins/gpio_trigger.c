/*

A simple c script that triggers a gpio pio for a specified amount of time after a delay

*/
// Imports the python library for converting into a python extension
#include <Python.h>

// Imports C libraries (both should be natively installed on the Raspbian image)
// More information on the wiringPi library can be found at wiringpi.com
#include <wiringPi.h>
#include <stdio.h>

//Sets the number of pin properties to be constant
const int N = 3;

// Sets up relevent pins (must be called before any gpio function)
int init(int pins[][N],int number_of_pins)
{
	/* Function Inputs:
		pins = an array consisting of 3 columns.
			Column 1 = the physical pin number
			Column 2 = whether the pin is set up for input (0) or output (1)
			Column 3 = (for input only -ignored for output) sets the pull up/down resistor to be off (0), pull down (1) or pull up (2)
		number_of_pins = the total number of pins passed to the function
	*/

	// Defines indexing variable
	int i;

	// Sets up GPIO pins with the physical pin naming convention.
	wiringPiSetupPhys();
	
	// Sets up each pin individually.	
	for(i = 0; i < number_of_pins; i++)
	{	
		// Sets the pin for input or output
		pinMode(pins[i][0],pins[i][1]);

		// If the pin is set up for input, sets the pull up/down resitor
 		if (pins[i][1] == 0)
		{
			pullUpDnControl(pins[i][0],pins[i][2]);
		}
	}
	
	// Returns 0 on success.
	return 0;
}


// Function that triggers GPIO pins
int trigger(float delay_time, float pulse_time, int pins[], int size)
{
	/* Function inputs:
		delay_time = the time delay (in microseconds) before triggering the pins
		pulse_time = the time (in microseconds) that the output signal will last
		pins = the pin numbers to be triggered
		size = the number of above pins
	*/

	// Defines indexing variable
	int i;

	// Delays the trigger
	delayMicroseconds(delay_time);

	// Triggers the pins
	for(i=0;i<size;i++)
	{
		digitalWrite (pins[i],HIGH);
	}

	// Leaves the pins active
	delayMicroseconds(pulse_time);

	// Deactivates pins
	for(i=0;i<size;i++)
	{
		digitalWrite (pins[i],LOW);
	}

	return 0;
}

// Function to read the current state of a single pin
int read(int pin)
{

	/* Function Inputs:
		pin = the pin number to be read
	*/

	return digitalRead(pin);
}


// A testing function that pulses on and off for a bit.
int pulse_test(int pin, float delay){

	// Defines inital variables
	int trigger_pin[1][3] = {{pin,1,0}};
	int k;

	// Initialises the trigger pin
	init(trigger_pin, N);

	// Pulses to access accuracy
	for(k = 0; k<100000; k++)
	{
		trigger(delay,delay,trigger_pin[0],N);	
	}

	return 0;
}


// Currently used to launch tests
int main(){

	pulse_test(8,10);

	return 0;
}
