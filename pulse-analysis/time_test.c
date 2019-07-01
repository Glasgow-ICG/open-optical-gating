/* A simple C script to test the distribution of wait times.
*/

// Imports relevent libraries
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

//A function that saves the array to a csv file
int save_array(int rows, int cols, float array[rows][cols], char filename[]){
	
	// Defines initial variables
	int n,m;

	// Opens the file for writing
	FILE *f = fopen(filename,"w");
	fclose(f);
	f = fopen(filename,"a");
	

	// Writes data to the file
	for (n=0;n<rows;n++){
		for (m=0;m<cols;m++){
			fprintf(f,"%f,",array[n][m]);
		}
		
		// Prints a new line for next data set
		fprintf(f,"\n");
	}
	
	// Closes the file
	fclose(f);

	return 0;

}

// Function that takes wait time and number of iterations as an input parameter and outputs distribution to a text file
int main(int argc, char *argv[]){

	// Checks argument count
	int arg_count =5;
	if ( argc < arg_count){
		printf("Too few arguments: expected number of itterations, start delay time (ns), end delay time (ns) and interval in delay times\n");
		return 1;
	}
	else if ( argc > arg_count){
		printf("Too few arguments: expected number of itterations, start delay time (ns), end delay time (ns) and interval in delay times\n");
		return 2;
	}

	// Defines initial variables
	int itterations = atoi(argv[1]);
	long wait_time_start = atoi(argv[2]);
	long wait_time_end = atoi(argv[3]);
	long wait_time_interval = atoi(argv[4]);
	long wait_time = wait_time_start;
	int i,j;
	int wait_time_size;
	struct timespec start, end;
	char filename[1000];

	//Calculates wait_time array size
	wait_time_size = (wait_time_end - wait_time_start)/wait_time_interval;

	//Initialises timing data array
	float time_ary[wait_time_size][itterations];


	// Loops over sleep parameters
	for (j=0;j<wait_time_size;j++){
		// Loop to perform the wait opperation
		for (i=0;i<itterations;i++){

			// Calculates the real wait time
			clock_gettime(CLOCK_MONOTONIC_RAW, &start);
			nanosleep((const struct timespec[]){{0, wait_time}}, NULL);
			clock_gettime(CLOCK_MONOTONIC_RAW, &end);
			
			// Assigns value to array
			time_ary[j][i] = (float)(end.tv_sec - start.tv_sec + (end.tv_nsec - start.tv_nsec)*1e-9);
		}
		
		// Increases wait time
		wait_time += wait_time_interval;
	}
	
	// Creates file name for csv file
	strcpy(filename,"hist-nanosleep");
	for (i=1;i<argc;i++){
		strcat(filename,"-");
		strcat(filename,argv[i]);
	}
	strcat(filename,".csv");


	// Saves result
	save_array(wait_time_size, itterations, time_ary,filename);

	return 0;
}

