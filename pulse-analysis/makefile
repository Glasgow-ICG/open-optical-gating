# A makefile for the timing test program in c

# Sets variables and flags
EXE=c_times
PREPROG=time_test.o
MAIN=time_test.c

OPTFLAGS=-O2
DEBUGFLAGS=-g
CFLAGS=-Wall


# Builds for debugging
debug: CFLAGS+=$(DEBUGFLAGS)
debug: $(EXE)

# Builds for release
opt: CFLAGS+=$(OPTFLAGS)
opt: $(EXE) 


# Builds the program
$(EXE): $(PREPROG)
	gcc $(CFLAGES) -o $@ $^

$(PREPROG): $(MAIN)
	gcc $(CFLAGS) -c $^


# Cleans the folder
clean:
	rm *.o
	rm $(EXE)
