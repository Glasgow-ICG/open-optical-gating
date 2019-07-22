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



// Sets up relevent pins (must be called before any gpio function)
static PyObject *fastpins_init(PyObject *self){

	// Sets up GPIO pins with the physical pin naming convention.
	wiringPiSetupPhys();

	return Py_BuildValue("i",0);
}

// Sets up an individual pin
static PyObject *fastpins_setpin(PyObject *self, PyObject *args){

	/* Function inputs:
		pin = the pin to be set up
		pin_value = the read (0) or write (1) value for the pin
		pin_pud = [read pins only] no resistor (0), pull up resistor (1) or pull down resistor (2)
	*/
	int pin, pin_value,pin_pud;

	// Convert python arguments into C integers
	if (!PyArg_ParseTuple(args,"iii",&pin, &pin_value, &pin_pud)){
		return NULL;
	}

	// Sets up the pin
	pinMode(pin,pin_value);

	// If the pin is read, sets the pud resistor value
	if (pin_value == 1){
		pullUpDnControl(pin,pin_pud);
	}

	return Py_BuildValue("i",0);

}
/// Function that triggers the laser and camera in pulse mode
static PyObject *fastpins_pulse(PyObject *self, PyObject *args){

	/* Function inputs:
		delay_time = the time delay (in microseconds) before triggering the pins
		pulse_time = the time (in microseconds) that the output signal will last
		laser_pin = the pin number of the laser trigger GPIO pin
		camera_pin = the pin number of the fluorescence camera GPIO pin
	*/

	// Defines argument variables
	float delay_time, pulse_time;
	int  laser_pin, camera_pin;

	// Converts arguments from python to C
	if (!PyArg_ParseTuple(args,"ffii", &delay_time, &pulse_time, &laser_pin, &camera_pin))
	{
		return NULL;
	}


	// Delays the trigger
	delayMicroseconds(delay_time);

	// Triggers the pins
	digitalWrite (camera_pin,HIGH);
	digitalWrite (laser_pin,HIGH);

	// Leaves the pins active
	delayMicroseconds(pulse_time);

	// Deactivates pins
	digitalWrite (laser_pin, LOW);
	digitalWrite (camera_pin,LOW);

	Py_RETURN_NONE;

}

/// Function that triggers the laser and camera in edge mode
static PyObject *fastpins_edge(PyObject *self, PyObject *args){

	/* Function inputs:
		delay_time = the time delay (in microseconds) before triggering the pins
		laser_pin = the pin number of the laser trigger GPIO pin
		camera_pin = the pin number of the fluorescence camera GPIO pin
		syncb_pin = the pin connected to the SYNC-B output (pin should be set up to read)
	*/

	// Defines argument variables
	float delay_time;
	int  laser_pin, camera_pin,syncb_pin;

	// Converts arguments from python to C
	if (!PyArg_ParseTuple(args,"fiii", &delay_time, &laser_pin, &camera_pin, &syncb_pin))
	{
		return NULL;
	}

	// Delays the trigger
	delayMicroseconds(delay_time);

	// Triggers the pins
	digitalWrite (camera_pin,HIGH);
	digitalWrite (laser_pin,HIGH);

	// Waits until capture is complete
	while (digitalRead(syncb_pin) == 1){}

	// Deactivates pins
	digitalWrite (laser_pin, LOW);
	digitalWrite (camera_pin,LOW);

	Py_RETURN_NONE;

}

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

	// Reads the pin value
	pin_read_value = digitalRead(pin);

	return Py_BuildValue("i",pin_read_value);
}


// Method Table
static PyMethodDef fastpinsMethods[] = {

	{"read",fastpins_read, METH_VARARGS},
	{"init",fastpins_init, METH_NOARGS},
	{"setpin",fastpins_setpin,METH_VARARGS},
	{"pulse",fastpins_pulse,METH_VARARGS},
	{"edge",fastpins_edge,METH_VARARGS},
	{NULL,NULL} /*Sentinel*/

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
