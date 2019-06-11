/*
 *  WorkThreads.h
 *  
 *	Class IntegerWorkThreads which manages the distribution of work over multiple threads
 *	See comments in associated file WorkThreads.cpp for more details.
 *
 */
#ifndef __WORKTHREADS_H__
#define __WORKTHREADS_H__

#include <stdio.h>
#include <pthread.h>
#include <unistd.h>
#include "jAssert.h"

enum
{
	kWorkFunctor = 1
};

class BaseProgressBar;
class JMutex;

#ifdef MAX_THREADS
	const int kAbsoluteMaxThreads = MAX_THREADS;
#else
	const int kAbsoluteMaxThreads = 16;
#endif

/*	If gMaxThreads is higher than the number we actually use (probably gRecommendedNumThreads),
	then we end up wasting  CPU time with unnecessary overheads when the idle threads
	wake up in response to the broadcast signal.			*/
extern int kMaxThreads, gUserSpecifiedMaxThreads, gRecommendedNumThreads;

// Magic numbers for threading work block size
enum
{
	kNoThreading = -1,		// Run everything on the main thread
	kSizeToRunInOneGo = -2	// Divide the work evenly among available threads,
							// so they don't have to request more work, but do
							// it all in one chunk
};

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

extern "C" void RunBlocks(dispatch_group_t group, void *obj, void *workBlock, long *firstItem, long *size, long threadsUsedForThisWorkMask, long maxThreads);
extern "C" void StaticBlockCallback(void *obj, void *workBlock, int thisItem, int numItems, int thisThreadNum);

class IntegerWorkThreads
{
  private:
	
	struct WorkBlock
	{
		int				workType;
		IntegerWorkFunctor *callbackFunctor;
		void			*sharedClientData;
		
		int				nextWorkItem;
		
		int				lastWorkItem;
		int				deltaWorkItem;
		int				preferredWorkSize;
		volatile int	finishedFlag;
		BaseProgressBar	*progressBar;
	};

  public:
	struct GCDContext
	{
		IntegerWorkThreads	*obj;
		WorkBlock			*workBlock;
		int					firstItem[kAbsoluteMaxThreads], size[kAbsoluteMaxThreads];
		int					threadsToUse[kAbsoluteMaxThreads];
	};
  private:
	struct IWTThreadParms
	{
		// These variables are set up at start of day
		pthread_t			theThread;
		IntegerWorkThreads	*theObject;
		int					threadNum;
		JMutex				*threadMutex;
		pthread_cond_t		*startSignal, *completionSignal;
#if CAN_USE_GRAND_CENTRAL
		dispatch_group_t	group;
#endif
		
		// Access to these variables should be protected by threadMutex
		int					runningAtLevel;
		int					reservedAtLevel;		// At what level was this thread first reserved? All idle threads have reservedAtLevel==0
		bool				topLevelThread, runningUnderGCD;

		/*	Access to these variables should be protected by threadMutex.
			They should ONLY be used for triggering a thread to START
			a piece of work. Hence they can be re-used for second-level
			threading without causing problems for top-level threading	*/
		int					firstWorkItem;
		int					firstWorkItemSize;
		WorkBlock			*workBlock;

	};
	friend void StaticBlockCallback(void *obj, void *workBlock, int thisItem, int numItems, int thisThreadNum);
	friend void StaticBlockCallback2(void *cxt, size_t thisItem);

	/*	The thread parms can be static, but see comments in structure definition.
		The start/completed signals are static (i.e. apply to all threads globally)
		which will mean that threads may be triggered when there hasn't been any change
		that affects them. I'll stick with this unless it becomes a problem, though	*/
	static pthread_cond_t	threadStartSignal, threadCompletedSignal;
	static IWTThreadParms	parms[kAbsoluteMaxThreads];
	static bool				threadParmsInited;
	static int				numInstancesRunning;
  
  protected:
	static JMutex	*threadMutex;

	static void		*StaticWorkThreadCallback(void *parms);
	int				GetNextWorkItem(WorkBlock *workBlock, int *outWorkSize, bool mutexHeld);
	void			VirtualWorkThreadCallback(WorkBlock *workBlock, int thisItem, int numItems, int thisThreadNum);
	virtual void	IntegerWorkCallback(int inWorkType, int thisItem, void *userData) { ALWAYS_ASSERT(0); }
	virtual bool	ThreadShouldExitEarly(int thisWorkType, IntegerWorkFunctor *thisFunctor) const;
	int				DoWork(int workType, int first, int last, int delta, BaseProgressBar *inBar, void *userData = NULL, int preferredBlockSize = 1, IntegerWorkFunctor *callbackFunctor = NULL);

  public:
					IntegerWorkThreads(void);
	virtual			~IntegerWorkThreads();

	void			DoWorkWithFunctor(IntegerWorkFunctor *functor, int first, int last, int delta, void *userData = NULL, BaseProgressBar *inBar = NULL);
	static void		CleanUpWorkThreads(void);
};

extern IntegerWorkThreads *gWorkThreads;

#endif
