//
//  JProgressWindow.mm
//  Simple Preview
//
//  Created by Jonathan Taylor on 04/03/2011.
//  Copyright 2011 Durham University. All rights reserved.
//

#import "JProgressWindow.h"


@implementation JProgressWindow

-(id)initAsIndeterminate
{
	if (!(self = [self initWithWindowNibName:@"CalibratingStages"]))
		return nil;
	[[self window] makeKeyAndOrderFront:nil];
	return self;
}

-(BOOL)usesThread { return indicator.usesThreadedAnimation; }
-(void)setUsesThread:(BOOL)flag { indicator.usesThreadedAnimation = flag; }
-(void)startAnimation { [indicator startAnimation:nil]; }

@end
