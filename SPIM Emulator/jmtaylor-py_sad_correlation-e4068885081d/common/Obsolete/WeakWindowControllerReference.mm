/*
 *  WeakWindowControllerReference.cpp
 *  Simple Preview
 *
 *  Created by Jonathan Taylor on 12/08/2011.
 *  Copyright 2011 Durham University. All rights reserved.
 *
 */

#import "WeakWindowControllerReference.h"
#import "jNotifications.h"

const NSString *WindowOwnerGoingAway = @"jonny.wwcr.window_owner_going_away";

@implementation WeakWindowControllerReference

-(id)initForOwner:(id)own
{
	if (![super init])
		return nil;
	owner = own;
	return self;
}

-(void)dealloc
{
	self.controller = nil;

	// Send notification that owner is going away
	SendImmediateNotificationOnThisThread(WindowOwnerGoingAway, self);	
	
	[super dealloc];
}

-(void)windowClosing:(NSNotification*)note
{
	CHECK(note.object == _controller);
	_controller = nil;
}

-(void)setController:(NSWindowController *)w
{
	[[NSNotificationCenter defaultCenter] removeObserver:self];
	_controller = w;
	if (w != nil)
		[[NSNotificationCenter defaultCenter] addObserver:self selector:@selector(windowClosing:) name:NSWindowWillCloseNotification object:_controller.window];
}

-(NSWindowController *)controller
{
	return _controller;
}

@end
