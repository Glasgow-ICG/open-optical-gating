//
//  jNotifications.h
//
//	Copyright 2010-2015 Jonathan Taylor. All rights reserved.
//
//	Utility functions relating to Cocoa notifications
//

#import <Cocoa/Cocoa.h>

@protocol FrameProtocol;

void SendImmediateNotificationForFrameOnThisThread(NSString *notificationName, id obj, id<FrameProtocol> frame);
void SendImmediateNotificationOnThisThread(NSString *notificationName, id obj);
void QueueNotificationForFrameOnMainThread(NSString *notificationName, id obj, id<FrameProtocol> frame, bool coalesce = false, NSPostingStyle style = NSPostASAP);
void QueueNotificationOnMainThread(NSString *notificationName, id obj, bool coalesce = false, NSPostingStyle style = NSPostASAP);
void QueueNotificationOnMainThread2(NSNotification *myNotification, bool coalesce = false, NSPostingStyle style = NSPostASAP);
void SendImmediateNotificationOnMainThread(NSNotification *myNotification);

extern NSString *CloseSheetsForTermination;
