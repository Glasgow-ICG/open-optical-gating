//
//  GeometryObjects.mm
//
//	Copyright 2010-2015 Jonathan Taylor. All rights reserved.
//
//	Obj-C objects to wrap an NSRect and an NSPoint.
//	The main purpose is to allow GUI code to bind to individual variables within the structure
//

#import "GeometryObjects.h"

@implementation JRect

+(JRect*)rectWithNSRect:(const NSRect)r
{
	return [[[JRect alloc] initWithRect:r] autorelease];
}

-(id)initWithRect:(const NSRect)r
{
	self.ns = r;
	return self;
}

-(JRect*)roundedToIntegers
{
    return [JRect rectWithNSRect:NSMakeRect(round(rect.origin.x), round(rect.origin.y), round(rect.size.width), round(rect.size.height))];
}

-(float)x { return rect.origin.x; }
-(float)y { return rect.origin.y; }
-(float)w { return rect.size.width; }
-(float)h { return rect.size.height; }
-(const NSRect)ns { return rect; }

-(void)setX:(float)val { rect.origin.x = val; }
-(void)setY:(float)val { rect.origin.y = val; }
-(void)setW:(float)val { rect.size.width = val; }
-(void)setH:(float)val { rect.size.height = val; }
-(void)setNs:(const NSRect)ns { rect = ns; }

-(int)everything { return 1; }

+(NSSet*)keyPathsForValuesAffectingValueForKey:(NSString*)inKey
{
	NSSet* set = [super keyPathsForValuesAffectingValueForKey:inKey];
	if ([inKey isEqualToString:@"ns"])
		set = [set setByAddingObjectsFromSet:[NSSet setWithObjects:@"x", @"y", @"w", @"h", nil]];
	else if ([inKey isEqualToString:@"everything"])
		set = [set setByAddingObjectsFromSet:[NSSet setWithObjects:@"ns", @"x", @"y", @"w", @"h", nil]];
	else
		set = [set setByAddingObjectsFromSet:[NSSet setWithObjects:@"ns", nil]];
	return set;
}

@end

@implementation JPoint2

+(JPoint2*)pointWithNSPoint:(const NSPoint)p
{
	return [[[JPoint2 alloc] initWithPoint:p] autorelease];
}

-(id)initWithPoint:(const NSPoint)p
{
	self.ns = p;
	return self;
}

-(float)x { return point.x; }
-(float)y { return point.y; }
-(const NSPoint)ns { return point; }

-(void)setX:(float)val { point.x = val; }
-(void)setY:(float)val { point.y = val; }
-(void)setNs:(const NSPoint)ns { point = ns; }

-(int)everything { return 1; }

+(NSSet*)keyPathsForValuesAffectingValueForKey:(NSString*)inKey
{
	NSSet* set = [super keyPathsForValuesAffectingValueForKey:inKey];
	if ([inKey isEqualToString:@"ns"])
		set = [set setByAddingObjectsFromSet:[NSSet setWithObjects:@"x", @"y", nil]];
	else if ([inKey isEqualToString:@"everything"])
		set = [set setByAddingObjectsFromSet:[NSSet setWithObjects:@"ns", @"x", @"y", nil]];
	else
		set = [set setByAddingObjectsFromSet:[NSSet setWithObjects:@"ns", nil]];
	return set;
}

@end
