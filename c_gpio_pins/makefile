# A makefile for the timing test program in c

# Sets variables and flags
EXE=gpio_test
PREPROG=gpio_trigger.o
MAIN=gpio_trigger.c

OPTFLAGS=-O2
DEBUGFLAGS=-g
CFLAGS=-Wall
LIBS=-lwiringPi

# Builds for debugging
debug: CFLAGS+=$(DEBUGFLAGS)
debug: $(EXE)

# Builds for release
opt: CFLAGS+=$(OPTFLAGS)
opt: $(EXE) 


# Builds the program
$(EXE): $(PREPROG)
	gcc $(CFLAGES) $(LIBS) -o $@ $^ 

$(PREPROG): $(MAIN)
	gcc $(CFLAGS) -c $^ 


# Cleans the folder
clean:
	rm *.o
	rm $(EXE)
