/*
 *  GeometryObjects.h
 *
 *	Copyright 2010-2015 Jonathan Taylor. All rights reserved.
 *
 *	Obj-C objects to wrap an NSRect and an NSPoint.
 *	The main purpose is to allow GUI code to bind to individual variables within the structure
 *
 */

@interface JRect : NSObject
{
	NSRect rect;
}

+(JRect*)rectWithNSRect:(const NSRect)r;
-(id)initWithRect:(const NSRect)r;
-(JRect*)roundedToIntegers;

@property (readwrite) float x;
@property (readwrite) float y;
@property (readwrite) float w;
@property (readwrite) float h;
@property (readwrite) const NSRect ns;
@property (readonly) int everything;	// Can be monitored using KVO to see if any variable changes

@end

@interface JPoint2 : NSObject
{
	NSPoint point;
}

+(JPoint2*)pointWithNSPoint:(const NSPoint)r;
-(id)initWithPoint:(const NSPoint)r;

@property (readwrite) float x;
@property (readwrite) float y;
@property (readwrite) const NSPoint ns;
@property (readonly) int everything;	// Can be monitored using KVO to see if any variable changes

@end

#import "GeometryObjectsC.h"
