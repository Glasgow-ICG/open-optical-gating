#include "WorkThreadsLite.h"
#include "ProgressBar.h"

void IntegerWorkThreads::DoWorkWithFunctor(IntegerWorkFunctor *callbackFunctor, int first, int last, int delta, void *userData, BaseProgressBar *progressBar)
{
	// Check parameters are sane
	ALWAYS_ASSERT(delta != 0);
	if (last - first > 0)
		ALWAYS_ASSERT(delta > 0);
	else if (last - first < 0)
		ALWAYS_ASSERT(delta < 0);

	dispatch_queue_t mainQueue = dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0);
	dispatch_apply((last + delta - first) / delta, mainQueue, ^(size_t i)
	{
		callbackFunctor->Callback(first + i * delta, userData);
		if (progressBar != NULL)
			progressBar->DeltaProgress(1.0);

	});
}

IntegerWorkThreads *gWorkThreads = new IntegerWorkThreads();
