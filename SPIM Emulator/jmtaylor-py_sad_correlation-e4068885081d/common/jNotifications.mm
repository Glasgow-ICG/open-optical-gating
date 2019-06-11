//
//  jNotifications.mm
//
//	Copyright 2010-2015 Jonathan Taylor. All rights reserved.
//
//	Utility functions relating to Cocoa notifications
//

#import "jNotifications.h"

NSString *CloseSheetsForTermination = @"jmt.CloseSheetsForTermination";

void SendImmediateNotificationOnThisThread(NSString *notificationName, id obj)
{
	// Post a Cocoa notification, with any notification callbacks being executed on the CURRENT thread.
	[[NSNotificationCenter defaultCenter] postNotificationName:notificationName
														object:obj];
}

void SendImmediateNotificationForFrameOnThisThread(NSString *notificationName, id obj, id<FrameProtocol> frame)
{
	// Post a Cocoa notification related to a frame object, with any notification callbacks being executed on the CURRENT thread.
	// This is generally a bad idea since the notification will occur on whatever the current
	// thread is, which is probably not what we want. (Remember that frames are handled on
	// several different threads at various stages in the pipeline).
	// In a few cases this function is useful, though
	[[NSNotificationCenter defaultCenter] postNotificationName:notificationName 
											object:obj
											userInfo:[NSDictionary dictionaryWithObjectsAndKeys:frame, @"frame", nil]];
}

// NOTE: There is actually a comment below this post: http://www.mikeash.com/pyblog/friday-qa-2010-01-08-nsnotificationqueue.html
// that suggests the following code is not safe - the notification might apparently not necessarily run on the main thread.
// Did not really manage to get any clarification from cocoa-dev about the strict letter of the specification here.
// However in practice it seems to work ok. I can't help feeling that if it did not at least serialize with respect to
// the main run loop, and maintain thread safety with respect to GUI operations, then Very Bad Things would happen for
// a lot of people out there...
void QueueNotificationOnMainThread(NSString *notificationName, id obj, bool coalesce, NSPostingStyle style)
{
	NSNotification *myNotification = [NSNotification notificationWithName:notificationName object:obj];
	QueueNotificationOnMainThread2(myNotification, coalesce, style);
}

void QueueNotificationForFrameOnMainThread(NSString *notificationName, id obj, id<FrameProtocol> frame, bool coalesce, NSPostingStyle style)
{
	NSNotification *myNotification = [NSNotification notificationWithName:notificationName 
														object:obj 
														userInfo:[NSDictionary dictionaryWithObjectsAndKeys:frame, @"frame", nil]];
	QueueNotificationOnMainThread2(myNotification, coalesce, style);
}

void QueueNotificationOnMainThread2(NSNotification *myNotification, bool coalesce, NSPostingStyle style)
{
	// Add notification to the event queue to be serviced whenever the event queue is ready to do so
	dispatch_async(dispatch_get_main_queue(), 
	^{
        if (coalesce)
        {
            // Caller wants coalescing to be enabled on this notification
            // Default behaviour seems to be to keep the OLDEST notification on the queue
            // I would much rather keep the NEWEST (most up-to-date information)!
            // In order to achieve that, I remove any existing notifications from the queue before posting this one
            [[NSNotificationQueue defaultQueue] dequeueNotificationsMatching:myNotification
                                                                coalesceMask:NSNotificationCoalescingOnName | NSNotificationCoalescingOnSender];
        }
		[[NSNotificationQueue defaultQueue]
				enqueueNotification:myNotification
				postingStyle:style
				coalesceMask:(coalesce ? NSNotificationCoalescingOnName|NSNotificationCoalescingOnSender : NSNotificationNoCoalescing)
				forModes:[NSArray arrayWithObject:NSRunLoopCommonModes]];		
	});
}

void SendImmediateNotificationOnMainThread(NSNotification *myNotification)
{
	// Trigger notification callbacks immediately, running on the main thread.
	dispatch_sync(dispatch_get_main_queue(), 
	^{
		[[NSNotificationCenter defaultCenter] postNotification:myNotification];
	});
}
