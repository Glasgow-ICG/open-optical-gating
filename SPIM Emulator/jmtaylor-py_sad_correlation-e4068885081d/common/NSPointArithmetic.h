/*
 *  NSPointArithmetic.h
 *
 *  Copyright 2011-2015 Jonathan Taylor. All rights reserved.
 *
 *  C++ operator-based code to allow direct arithmetic of Cocoa's NSPoint objects
 *
 */

inline NSPoint operator+(NSPoint a, NSPoint b)
{
	NSPoint result = { a.x + b.x, a.y + b.y };
	return result;
}

inline NSPoint operator-(NSPoint a, NSPoint b)
{
	NSPoint result = { a.x - b.x, a.y - b.y };
	return result;
}

inline NSPoint operator/(NSPoint a, float b)
{
	NSPoint result = { a.x / b, a.y / b };
	return result;
}

inline NSPoint operator*(NSPoint a, float b)
{
	NSPoint result = { a.x * b, a.y * b };
	return result;
}

inline NSPoint operator+(NSPoint a, NSSize b)
{
	NSPoint result = { a.x + b.width, a.y + b.height };
	return result;
}
