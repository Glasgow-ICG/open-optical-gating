/*

A simple c script that triggers a gpio pio for a specified amount of time after a delay

*/
#ifndef STARTUP
#define STARTUP
#include <wiringpi.h>
#include <stdio.h>

wiringPiSetup();
pinMode(0, OUTPUT);

#endif


// Main function that triggers GPIO pin
int main(int argc, char *argv[])
{
	// Defines delay timings
	float delay_time = atoi(argv[1]);
	float pulse_time = atoi(argv[2]);

	// Pulses the GPIO pin after delay
	delay (delay_time);
	digitalWrite (0,HIGH);
	delay (pulse_time);
	digitalWrite (0,LOW);

	return 0;
}
