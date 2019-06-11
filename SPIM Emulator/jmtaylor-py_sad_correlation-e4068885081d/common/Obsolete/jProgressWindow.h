//
//  JProgressWindow.h
//  Simple Preview
//
//  Created by Jonathan Taylor on 04/03/2011.
//  Copyright 2011 Durham University. All rights reserved.
//

#import <Cocoa/Cocoa.h>


@interface JProgressWindow : NSWindowController {
	IBOutlet NSProgressIndicator *indicator;
}

-(id)initAsIndeterminate;
-(void)startAnimation;
@property BOOL usesThread;

@end
