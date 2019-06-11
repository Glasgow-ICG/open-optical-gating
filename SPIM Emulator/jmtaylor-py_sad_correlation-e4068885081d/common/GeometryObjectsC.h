/*
 *  GeometryObjectsC.h
 *
 *	Copyright 2010-2015 Jonathan Taylor. All rights reserved.
 *
 *
 */

#ifndef __GEOMETRY_OBJECTS_C_H__
#define __GEOMETRY_OBJECTS_C_H__ 1

struct IntegerPoint
{
	int x, y;
	IntegerPoint() : x(0), y(0) { }
	IntegerPoint(int a, int b) : x(a), y(b) { }
};

// Sadly I can't seem to get pass-by-reference to work with this - some of my ObjC property-based
// code doesn't compile if I pass a and b by reference. Not sure if this would be fix-able, but
// I'm just going to leave it as-is for now.
inline bool operator!=(IntegerPoint a, IntegerPoint b) { return ((a.x != b.x) || (a.y != b.y)); }

#endif
