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

NSBitmapImageRep *RawBitmapFromImage(const NSImage *image);
NSBitmapImageRep *RawBitmapFromImagePath(NSString *imagePath);
bool /*sizes matched*/ PopulateArrayFromBitmap(const NSBitmapImageRep *bitmap, double *destArray, int destWidth, int destHeight, int downsampleFactor, bool assertOnSizeMismatch);
void CopyNSImageToGWorld(const NSImage *image, GWorldPtr gWorldPtr, const CGRect *cropRect, double gain);
NSPoint FractionalCoordWithinImageView(const NSPoint &thePoint, const NSImageView *theView);
NSPoint ImageViewCoordToImageCoord(const NSPoint &thePoint, const NSImageView *theView);
NSPoint ImageCoordToImageViewCoord(const NSPoint &thePoint, const NSImageView *theView);
void BrightenNSImage(NSImage *image, int factor);
NSImage *TintImage(NSImage *srcImage, NSColor *tint, NSColor *saturation, double exposureOnScreen);
NSBitmapImageRep *TintBitmap(NSBitmapImageRep *srcBitmap, NSColor *tint, NSColor *saturation, double exposureOnScreen);
void AddNoiseToImage(NSImage *image, double level);

#endif
