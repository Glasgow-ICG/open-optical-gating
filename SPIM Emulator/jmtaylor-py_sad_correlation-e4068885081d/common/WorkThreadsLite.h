/*
 *  WorkThreads.h
 *  
 *	Class IntegerWorkThreads which manages the distribution of work over multiple threads
 *	See comments in associated file WorkThreads.cpp for more details.
 *
 */
#ifndef __WORKTHREADS_LITE_H__
#define __WORKTHREADS_LITE_H__

#include <stdio.h>
#include <pthread.h>
#include <unistd.h>
#include "jAssert.h"

class BaseProgressBar;
class JMutex;

class IntegerWorkFunctor
{
  public:
	virtual ~IntegerWorkFunctor() { }
	virtual void Callback(int index, void *userData) const = 0;
};

// Macro used to define functors providing work thread callbacks to client code
#define DEFINE_INTEGER_WORK_FUNCTOR(CLASS, METHOD, TYPE, NAME, USER_DATA_TYPE) \
class TYPE : public IntegerWorkFunctor\
{ \
CLASS	* theClass; \
public: \
TYPE(CLASS * in) : theClass(in) { } \
virtual ~TYPE() { } \
virtual void Callback(int index, void *userData) const \
{ \
return theClass->METHOD(index, (USER_DATA_TYPE *)userData); \
} \
}; \
TYPE NAME;

class IntegerWorkThreads
{
  public:
					IntegerWorkThreads(void) { }
	virtual			~IntegerWorkThreads() { }

	void			DoWorkWithFunctor(IntegerWorkFunctor *functor, int first, int last, int delta, void *userData = NULL, BaseProgressBar *inBar = NULL);
};

extern IntegerWorkThreads *gWorkThreads;

#endif
