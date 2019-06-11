//
//  BoundsRect.h
//
//	Copyright 2014-2015 Jonathan Taylor. All rights reserved.
//
//	Structure representing a rectangle with integer corner positions
//

#ifndef __BOUNDS_RECT_H__
#define __BOUNDS_RECT_H__

struct BoundsRect
{
	int x, y, w, h;
	BoundsRect() { x = y = w = h = 0; }
	BoundsRect(int inX, int inY, int inW, int inH) : x(inX), y(inY), w(inW), h(inH) { }
	BoundsRect(const BoundsRect &a) : x(a.x), y(a.y), w(a.w), h(a.h) { }
#if __OBJC__
	BoundsRect(const NSRect &a) : x(int(a.origin.x)), y(int(a.origin.y)), w(int(a.size.width)), h(int(a.size.height)) { }
	NSRect AsNSRect(void) const { return NSMakeRect(x, y, w, h); }
#endif
};

#endif
