/*
 *	jAssert.cpp
 *
 *  Copyright 2011-2015 Jonathan Taylor. All rights reserved.
 *
 *  Code to handle assertions
 */

#include "jAssert.h"
#include "assert.h"
#include "DebugPrintf.h"

static BaseAssertionHandler defaultHandler;
BaseAssertionHandler *assertionHandler = &defaultHandler;

void BaseAssertionHandler::AssertionFailed(int line, const char *function, const char *file)
{
	// Report the error
	// This code was moved out of assertion macro for code brevity and to make modification easier
	DebugPrintfFatal("An assertion was failed and the program has crashed", "Assertion failed on line %d, function %s, file %s\n", line, function, file);
	PullDownCode();
	// Included to satisfy the compiler, which wants to see unambiguously that this function will never return
	assert(false);
}

void BaseAssertionHandler::PullDownCode(void)
{
	/*	We are going to force an instant crash in order to trigger a break in the debugger (if present)
		As a result we need to flush buffers first - otherwise the message about the assertion may
		never show up!	*/
	fflush(stdout);
	fflush(stderr);
	
	// Now trigger the crash by dereferencing a null pointer
	// Note that the analyzer doesnt like this, so we hide it from the analyzer
#ifndef __clang_analyzer__
	*((volatile int *)0L) = 0;
#endif
	// Included to satisfy the compiler, which wants to see unambiguously that this function will never return
	assert(false);
}

bool BaseAssertionHandler::CheckCondition(bool condition, int line, const char *function, const char *file)
{
	if (!condition)
	{
		// Report the error
		DebugPrintf("Check failed on line %d, function %s, file %s\n", line, function, file);
	}
	return condition;
}
