//
//  SetValueButton.mm
//  Simple Preview
//
//  Created by Jonathan Taylor on 02/03/2011.
//  Copyright 2011 Durham University. All rights reserved.
//

#import "SetValueButton.h"


@implementation SetValueButton

-(IBAction)setIntValue:(id)sender;
{
	[dest takeIntValueFrom:src];
}

-(IBAction)setDoubleValue:(id)sender;
{
	[dest takeIntValueFrom:src];
}

@end

@implementation SetValueController

-(IBAction)setIntValue:(id)sender;
{
	[dest takeIntValueFrom:src];
}

-(IBAction)setDoubleValue:(id)sender;
{
	[dest takeIntValueFrom:src];
}

@end
