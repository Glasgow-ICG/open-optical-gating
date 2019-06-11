//
//	jDispatchTimer.h
//
//	Copyright 2010-2015 Jonathan Taylor. All rights reserved.
//
//	Cocoa class wrapping a timer, which will execute a block on a GCD queue after a specified interval
//

#import "jDispatchTimer.h"

@interface JDispatchTimer()
    @property (readwrite) double oneoffTimeDue;
@end

@implementation JDispatchTimer

-(void)startTimer		// private
{
//	printf("Start timer %p, source %p with %lld %lld\n", self, timerSource, intervalNs, repeatIntervalNs);
    self.oneoffTimeDue = GetTime() + intervalNs * 1e-9;
	dispatch_source_set_timer(timerSource, dispatch_time(DISPATCH_TIME_NOW, intervalNs), repeatIntervalNs, flexibilityNs);
}

-(void)fireOneShotTimerNow
{
    CHECK(repeatIntervalNs == DISPATCH_TIME_FOREVER);     // I have only implemented this for one-shot timers
    self.oneoffTimeDue = GetTime();
	dispatch_source_set_timer(timerSource, dispatch_time(DISPATCH_TIME_NOW, 0.0), repeatIntervalNs, flexibilityNs);
}

#if 0
// Debug code that can be useful for tracking down retain/release issues.
-(id)retain
{
//	printf("JDispatchTimer retain %p (will be %d)\n", self, self.retainCount+1);
	id result;
	@synchronized(self)
	{
	NSArray *symbols = [NSThread callStackSymbols];
	NSLog(@"JDispatchTimer retain %p (will be %d)\n", self, self.retainCount+1);
	NSLog(@"%@", symbols);
	result = [super retain];
	}
	return result;
}
-(void)release
{
//	printf("JDispatchTimer release %p (will be %d)\n", self, self.retainCount-1);
	@synchronized(self)
	{
		NSArray *symbols = [NSThread callStackSymbols];
		printf("JDispatchTimer release %p (will be %d)\n", self, self.retainCount-1);
		NSLog(@"%@", symbols);
		[super release];
	}
}
#endif

-(id)initForQueue:(dispatch_queue_t)queue withInterval:(double)dt flex:(double)flexibility repeat:(bool)repeat timeCritical:(bool)timeCritical withHandler:(dispatch_block_t)handler
{
	// Initialize a timer object running on the specified GCD queue, that will execute the block 'handler' when it fires
	if (!(self = [super init]))
		return nil;
	
	firedOneShot = false;	
    timerSource = dispatch_source_create(DISPATCH_SOURCE_TYPE_TIMER, 0, timeCritical ? DISPATCH_TIMER_STRICT : 0, queue);
	if (timerSource == NULL)
	{
		/*	Not all OS versions support DISPATCH_TIMER_STRICT, and there's no totally obvious way of
			knowing which do and which do not (although I suspect it is first supported on 10.9.something).
			As a result, if the first call fails then I just retry with no flags	*/
		timerSource = dispatch_source_create(DISPATCH_SOURCE_TYPE_TIMER, 0, 0, queue);
	}
	ALWAYS_ASSERT(timerSource != NULL);
	
	if (repeat)
	{
		dispatch_source_set_event_handler(timerSource, handler);
	}
	else
	{
		dispatch_block_t handlerCopy = Block_copy(handler);		// Take a copy of the block to be sure it doesn't go out of scope before we use it
//		printf("Init timer %p with handler copy %p of original %p\n", self, (void*)handlerCopy, (void*)handler);

		dispatch_source_set_cancel_handler(timerSource, ^{
//			printf("Dispatch source cancellation handler called for %p, source %p\n", self, timerSource);
			Block_release(handlerCopy);
		});
		dispatch_source_set_event_handler(timerSource, ^{
 			if (firedOneShot)
			{
				/*	I currently don't have a solution to the problem where a call to restartOneShotTimer can
					occur after the timer has *actually* fired, but before this handler has been called.
					This seems to be a limitation of the API. For now I am just working around it by
					ignoring a second fire of the same timer	*/
				printf("WARNING - one shot timer fired more than once (known API issue). Ignoring\n");
			}
			else
			{
//				printf("One shot timer %p fired - should be on queue %p\n", self, queue);
				handlerCopy();
				// We do not release handlerCopy here - we do that in the cancel handler
				firedOneShot = true;
				/*	If we get here then the user must consider the timer as "dead" and the timer should go away of its own accord
					Unfortunately there is an implicit 'retain' of self due to the handler block which will have been retained
					during the call to dispatch_source_set_event_handler. We must explicitly cancel the dispatch source in order
					for our own 'release' calls to take the retain count down to zero	*/
//				printf("%p cancel %p\n", self, timerSource);
				dispatch_source_cancel(timerSource);

				[self release];
			}
		});
	}
	intervalNs = (uint64_t)(dt * NSEC_PER_SEC);
	flexibilityNs = (uint64_t)(flexibility * NSEC_PER_SEC);
	repeatIntervalNs = repeat ? intervalNs : DISPATCH_TIME_FOREVER;

	[self startTimer];
	dispatch_resume(timerSource);

	return self;
}

+(id)newOneShotTimerOnQueue:(dispatch_queue_t)queue afterInterval:(double)dt flex:(double)flexibility critical:(bool)critical withHandler:(dispatch_block_t)handler
{
	// Caller gets a retained object that they must release from their one shot callback.
	// Note that we do an *extra* retain here to balance the release that we do from our own event hander above.
	return [[[JDispatchTimer alloc] initForQueue:queue withInterval:dt flex:(double)flexibility repeat:false timeCritical:critical withHandler:handler] retain];
}

+(id)oneShotTimerOnQueue:(dispatch_queue_t)queue afterInterval:(double)dt flex:(double)flexibility critical:(bool)critical withHandler:(dispatch_block_t)handler
{
	// Will release itself after firing
	return [[JDispatchTimer alloc] initForQueue:queue withInterval:dt flex:(double)flexibility repeat:false timeCritical:critical withHandler:handler];
}

+(id)allocRepeatingTimerOnQueue:(dispatch_queue_t)queue atInterval:(double)dt flex:(double)flexibility critical:(bool)critical withHandler:(dispatch_block_t)handler
{
    return [[JDispatchTimer alloc] initForQueue:queue withInterval:dt flex:(double)flexibility repeat:true timeCritical:critical withHandler:handler];
}

-(void)suspend
{
	// Suspend firing of the timer (it will not fire until -restart is called)
//	printf("Suspend %p\n", self);
	ALWAYS_ASSERT(repeatIntervalNs != DISPATCH_TIME_FOREVER);		// Not supported for one-shot fire-and-forget timers
	dispatch_source_set_timer(timerSource, DISPATCH_TIME_FOREVER, DISPATCH_TIME_FOREVER, 0);
}

-(void)restart
{
	// Resume repeated firing after a previous call to -suspend.
//	printf("Restart %p (source %p)\n", self, timerSource);
	ALWAYS_ASSERT(repeatIntervalNs != DISPATCH_TIME_FOREVER);		// Not supported for one-shot fire-and-forget timers
	[self startTimer];
}

-(void)adjustNextInterval:(double)newInterval
{
//	printf("Adjust %p, interval %lf\n", self, newInterval);
	[self suspend];
	intervalNs = repeatIntervalNs = (uint64_t)(newInterval * NSEC_PER_SEC);
	[self startTimer];
}

-(void)restartOneShotTimer
{
    [self restartOneShotTimerWithIntervalInNsFromNow:intervalNs];
}

-(void)restartOneShotTimerWithIntervalInSecsFromNow:(double)newIntervalFromNow
{
    [self restartOneShotTimerWithIntervalInNsFromNow:uint64_t(newIntervalFromNow * 1e9)];
}

-(void)restartOneShotTimerWithIntervalInNsFromNow:(uint64_t)newIntervalNs
{
	// This has a different name as a reminder that it must only be called *before* the timer has fired.
	// A one shot timer cannot be restarted when it has already fired
	// Note that thought is required here from the caller - the restart must occur on the same queue that
	// the callback will run on, or window conditions are possible.
//	printf("restart one-shot %p (source %p) at interval %lfs from now\n", self, timerSource, newIntervalNs*1e-9);
	ALWAYS_ASSERT(!firedOneShot);
	if (timerSource != nil)
	{
		dispatch_source_set_timer(timerSource, DISPATCH_TIME_FOREVER, DISPATCH_TIME_FOREVER, 0); // Suspend momentarily while we mess with things
        intervalNs = newIntervalNs;
		[self startTimer];
	}
	else
		ALWAYS_ASSERT(0);
}

-(void)cancel
{
//	printf("Cancel timer %p (source %p)\n", self, timerSource);
	if (!dispatch_source_testcancel(timerSource))
	{
		dispatch_source_cancel(timerSource);
		dispatch_release(timerSource);
		[self release];
	}
}

-(void)dealloc
{
//	printf("Dealloc timer %p (source %p)\n", self, timerSource);
	
	if (!dispatch_source_testcancel(timerSource))
	{
		/*	I'm pretty sure we shouldn't ever get here - we self-retain and release when
			the callback is called or when it is cancelled. Just to be safe I have included
			this code though. Note however that since I am not sure why we would ever get here,
			I am relucant to call dispatch_release as well...	*/
		printf("%p dealloc apparently without cancelling %p\n", self, timerSource);
		CHECK(0);
		dispatch_source_cancel(timerSource);
	}
	
	[super dealloc];
//	printf("Done dealloc timer %p\n", self);
}

@synthesize oneoffTimeDue = _oneoffTimeDue;

@end
