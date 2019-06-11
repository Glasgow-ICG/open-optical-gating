//
//  SetValueButton.h
//  Simple Preview
//
//  Created by Jonathan Taylor on 02/03/2011.
//  Copyright 2011 Durham University. All rights reserved.
//

#import <Cocoa/Cocoa.h>


@interface SetValueButton : NSButton {
	IBOutlet NSControl *src;
	IBOutlet NSControl *dest;
}

-(IBAction)setIntValue:(id)sender;
-(IBAction)setDoubleValue:(id)sender;

@end

@interface SetValueController : NSObject {
	IBOutlet NSControl *src;
	IBOutlet NSControl *dest;
}

-(IBAction)setIntValue:(id)sender;
-(IBAction)setDoubleValue:(id)sender;

@end
