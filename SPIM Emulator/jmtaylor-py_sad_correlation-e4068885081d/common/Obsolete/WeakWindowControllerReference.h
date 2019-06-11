/*
 *  WeakWindowControllerReference.h
 *  Simple Preview
 *
 *  Created by Jonathan Taylor on 12/08/2011.
 *  Copyright 2011 Durham University. All rights reserved.
 *
 */

extern const NSString *WindowOwnerGoingAway;

@interface WeakWindowControllerReference : NSObject
{
  @private
	NSWindowController	*_controller;
	id					owner;
}

-(id)initForOwner:(id)own;
@property (nonatomic, assign) NSWindowController *controller;

@end
