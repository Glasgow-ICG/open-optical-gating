/*
 *  jTimeUtils.cpp
 *  
 *	Copyright 2010-2015 Jonathan Taylor. All rights reserved.
 *
 *	Utilities to determine elapsed time
 *
 */

#include "jTimeUtils.h"
#include <stdio.h>
#include <sys/time.h>
#include <sys/resource.h>
#if __MACH__
	#include <CoreServices/CoreServices.h>
#endif

double GetTimeAbsolute(void)
{
#if __MACH__
	// I think this ought to give better resolution on OS X:
	// Note that UpTime is deprecated - if I want to faff with replacing it then
	// I can look at mach_absolute_time and mach_timebase_info... or just use gettimeofday!
	Nanoseconds ns = AbsoluteToNanoseconds(UpTime());
	return ns.lo * 1e-9 + ns.hi * (1e-9 * (1LL<<32));
#else
	// Standard BSD function
	struct timeval t;
	struct timezone tz;

	gettimeofday(&t, &tz);

	return t.tv_sec + t.tv_usec * 1e-6;
#endif
}

// Global variable for program launch time
static double gStartTime = GetTimeAbsolute();

double GetTimebaseStart(void) { return gStartTime; }

double GetTime(void)
{
	// Subtract the start time for convenience. Can't think of any circumstance where I'd care about an absolute time.
	return GetTimeAbsolute() - gStartTime;
}

void ProcessorTime(double *userTime, double *sysTime, double *bothTime)
{
	// Utility function to return elapsed actual execution time (for performance monitoring)
	// Note that on some linux systems [was certainly true for the cray...], thread time is allocated to "children".
	// We need to include that time as well in our calculation
	struct rusage self, children;
	
	getrusage(RUSAGE_SELF, &self);
	getrusage(RUSAGE_CHILDREN, &children);
	
	struct timeval t_self1 = self.ru_utime;
	struct timeval t_children1 = children.ru_utime;
	struct timeval t_self2 = self.ru_stime;
	struct timeval t_children2 = children.ru_stime;
	
	// t stores a value for seconds and a value for microseconds.
	// I need to convert this to a very large value for microseconds.
	unsigned long long user_us = (unsigned long long)t_self1.tv_sec * 1000000LL + t_self1.tv_usec + t_children1.tv_sec * 1000000LL + t_children1.tv_usec;
	unsigned long long sys_us = (unsigned long long)t_self2.tv_sec * 1000000LL + t_self2.tv_usec + t_children2.tv_sec * 1000000LL + t_children2.tv_usec;

	if (userTime != NULL)
		*userTime = user_us / 1e6;
	if (sysTime != NULL)
		*sysTime = sys_us / 1e6;
	if (bothTime != NULL)
		*bothTime = (user_us + sys_us) / 1e6;
}

double CalcElapsedSecs(double t1, double t2)
{
	// Kind of trivial utility function, not sure quite why I bothered with this!
	return t2 - t1;
}

void ReportElapsedTime(double t1, double t2, const char *theText)
{
	printf("%s: %.02lfs\n", theText, t2 - t1);
}

void ReportElapsedTime_us(double t1, double t2, const char *theText)
{
	printf("%s: %.02lfus\n", theText, CalcElapsedSecs(t1, t2) * 1e6);
}

void PauseFor(double secs)
{
	// Block for the specified number of seconds before returning
	struct timeval timeout;
	timeout.tv_sec = (int)secs;
	timeout.tv_usec = (int)((secs - (int)secs) * 1000000);
	select(0, NULL, NULL, NULL, &timeout);
}
