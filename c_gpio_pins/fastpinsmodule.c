/*

A simple c script that triggers a gpio pio for a specified amount of time after a delay

*/
// Inlcudes python header for python extension
#define PY_SSIZE_T_CLEAN
#include <Python.h>

// Imports C libraries (both should be natively installed on the Raspbian image)
// More information on the wiringPi library can be found at wiringpi.com
#include <wiringPi.h>
#include <stdio.h>

//
///Sets the number of pin properties to be constant
//const int N = 3;
//
/// Sets up relevent pins (must be called before any gpio function)
//int init(int pins[][N],int number_of_pins)
//
//       /* Function Inputs:
//       	pins = an array consisting of 3 columns.
//       		Column 1 = the physical pin number
//       		Column 2 = whether the pin is set up for input (0) or output (1)
//       		Column 3 = (for input only -ignored for output) sets the pull up/down resistor to be off (0), pull down (1) or pull up (2)
//       	number_of_pins = the total number of pins passed to the function
//       */
//
//       // Defines indexing variable
//       int i;
//
//       // Sets up GPIO pins with the physical pin naming convention.
//       wiringPiSetupPhys();
//       
//       // Sets up each pin individually.	
//       for(i = 0; i < number_of_pins; i++)
//       {	
//       	// Sets the pin for input or output
//       	pinMode(pins[i][0],pins[i][1]);
//
//       	// If the pin is set up for input, sets the pull up/down resitor
//		if (pins[i][1] == 0)
//       	{
//       		pullUpDnControl(pins[i][0],pins[i][2]);
//       	}
//       }
//       
//       // Returns 0 on success.
//       return 0;
//
//
//
/// Function that triggers GPIO pins
//static PyObject *fastpins_trigger( PyObject *self, PyObject *args)
//
//       /* Function inputs:
//       	delay_time = the time delay (in microseconds) before triggering the pins
//       	pulse_time = the time (in microseconds) that the output signal will last
//       	pins = the pin numbers to be triggered
//       	size = the number of above pins
//       */
//       
//       // Defines argument variables
//       float delay_time, pulse_time;
//       int pins[], size;
//
//       // Converts arguments from python to C
//       if (!PyArgParseTuple(args,"ffii", &delay_time, &pulse_time, &pins, &size))
//       {
//       	return NULL;
//       }
//
//       // Defines indexing variable
//       int i;
//
//       // Delays the trigger
//       delayMicroseconds(delay_time);
//
//       // Triggers the pins
//       for(i=0;i<size;i++)
//       {
//       	digitalWrite (pins[i],HIGH);
//       }
//
//       // Leaves the pins active
//       delayMicroseconds(pulse_time);
//
//       // Deactivates pins
//       for(i=0;i<size;i++)
//       {
//       	digitalWrite (pins[i],LOW);
//       }
//
//       Py_RETURN_NONE;
//
//
//

// Function to read the current state of a single pin
static PyObject *fastpins_read(PyObject *self, PyObject *args)
{

	/* Function Inputs:
		pin = the pin number to be read
	*/
	int pin;
	int pin_read_value;

	// Converts arguments from python to C
	if (!PyArg_ParseTuple(args, "i", &pin))
		return NULL;

	// Sets up the pin (until init function is included)
	wiringPiSetupPhys();
	pinMode(pin,0);

	// Reads the pin value
	pin_read_value = digitalRead(pin);
	
	return Py_BuildValue("i",pin_read_value);
}


// Method Table
static PyMethodDef fastpinsMethods[] = {

	{"read",fastpins_read, METH_VARARGS,
	"Reads a pin value"},

	{NULL,NULL,0,NULL} /*Sentinel*/

};

// Module definition structure
static struct PyModuleDef fastpinsmodule = {
	PyModuleDef_HEAD_INIT,
	"fastpins",	// Module name
	NULL,		// Module documentation
	-1,		// -1 keeps state in global variables
	fastpinsMethods
};
// Initialisation function
PyMODINIT_FUNC PyInit_fastpins(void)
{
	return PyModule_Create(&fastpinsmodule);
}
