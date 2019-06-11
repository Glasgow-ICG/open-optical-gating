/*
 *  jCocoaImageUtils.h
 *
 *  Copyright 2011-2015 Jonathan Taylor. All rights reserved.
 *
 *	Utility functions for working with images under Cocoa
 *
 */

#ifndef __J_COCOA_IMAGE_UTILS_H__
#define __J_COCOA_IMAGE_UTILS_H__ 1

bool /*sizes matched*/ PopulateArrayFromBitmap(const char *bitmapPath, double *destArray, int destWidth, int destHeight, int downsampleFactor, bool assertOnSizeMismatch);

#endif
