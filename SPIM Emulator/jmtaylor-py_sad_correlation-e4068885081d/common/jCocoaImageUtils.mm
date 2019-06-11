/*
 *  jCocoaImageUtils.mm
 *
 *  Copyright 2011-2015 Jonathan Taylor. All rights reserved.
 *
 *	Utility functions for working with images under Cocoa
 *
 */

#import <Cocoa/Cocoa.h>
#include "jAssert.h"
#import "jCocoaImageUtils.h"
#import "NSPointArithmetic.h"
#include "jCommon.h"

bool /*sizes matched*/ PopulateArrayFromBitmap(const NSBitmapImageRep *bitmap, double *destArray, int destWidth, int destHeight, int downsampleFactor, bool assertOnSizeMismatch)
{
    // Handle potential difference in bitmap and array dimensions
    bool sizeMismatch = false;
    if (bitmap.pixelsWide != destWidth * downsampleFactor)
        sizeMismatch = true;
    if (bitmap.pixelsHigh != destHeight * downsampleFactor)
        sizeMismatch = true;
    if (assertOnSizeMismatch)
        ALWAYS_ASSERT(!sizeMismatch);
    int width = MIN(destWidth, (int)bitmap.pixelsWide/downsampleFactor);
    int height = MIN(destHeight, (int)bitmap.pixelsHigh/downsampleFactor);
    int x0 = 0, y0 = 0;
    if (sizeMismatch)
    {
        if (bitmap.pixelsWide / downsampleFactor < destWidth)
            x0 = (destWidth - (int)bitmap.pixelsWide / downsampleFactor) / 2;
        if (bitmap.pixelsHigh / downsampleFactor < destHeight)
            y0 = (destHeight - (int)bitmap.pixelsHigh / downsampleFactor) / 2;
    }
    
    // For now we expect an 8- or 16-bit greyscale image
    if (bitmap.bitsPerPixel == 16)
    {
        for (int y = 0; y < height; y++)
        {
            const unsigned short *rowSource = (const unsigned short*)(bitmap.bitmapData + (y * downsampleFactor) * bitmap.bytesPerRow);
            ALWAYS_ASSERT((y+y0)*destWidth+(width+x0) <= destWidth * destHeight);
            double scaleFactor = 1.0 / 65536.0;
            for (int x = 0; x < width; x++)
            {
                destArray[(y+y0)*destWidth+(x+x0)] = rowSource[x*downsampleFactor] * scaleFactor;
            }
        }
    }
    else
    {
        ALWAYS_ASSERT(bitmap.bitsPerPixel == 8);
    //    ALWAYS_ASSERT(!(bitmap.bitmapFormat & NSAlphaFirstBitmapFormat));

        for (int y = 0; y < height; y++)
        {
            const unsigned char *rowSource = bitmap.bitmapData + (y * downsampleFactor) * bitmap.bytesPerRow;
            if (!CHECK((y+y0)*destWidth+(width+x0) <= destWidth * destHeight))
            {
                printf("%d %d %d %d %d %d\n", y, y0, x0, width, destWidth, destHeight);
            }
            ALWAYS_ASSERT((y+y0)*destWidth+(width+x0) <= destWidth * destHeight);
            double scaleFactor = 1.0 / 256.0;
            for (int x = 0; x < width; x++)
            {
                destArray[(y+y0)*destWidth+(x+x0)] = rowSource[x*downsampleFactor] * scaleFactor;
            }
        }
    }
    
    return sizeMismatch;
}

bool /*sizes matched*/ PopulateArrayFromBitmap(const char *bitmapPath, double *destArray, int destWidth, int destHeight, int downsampleFactor, bool assertOnSizeMismatch)
{
    NSAutoreleasePool *pool = [NSAutoreleasePool new];
    bool result = PopulateArrayFromBitmap(RawBitmapFromImagePath([SWF:@"%s", bitmapPath]), destArray, destWidth, destHeight, downsampleFactor, assertOnSizeMismatch);
    [pool drain];
    return result;
}

NSBitmapImageRep *RawBitmapFromImage(const NSImage *image)
{
	// Returns an NSBitmapImageRep from the NSImage that is passed in.
	// This isn't intended to handle all possible scenarios, but will do the best it can,
	// and assert if things don't make sense.
	// In practice most images I work with seem to contain one and only one bitmap image.
	if (!CHECK(image != nil))
		return nil;
		
	NSBitmapImageRep	*result = nil;
    NSArray				*repArray = [image representations];
	
	/*	n.b. a better way of implementing all this under 10.6 would actually be based on:
		 [[NSBitmapImageRep alloc] initWithCGImage:[image CGImageForProposedRect?:context:hints:]]
		 See http://cocoadev.com/index.pl?NSBitmapImageRep	*/
		 
    for (size_t imgRepresentationIndex = 0; imgRepresentationIndex < repArray.count; ++imgRepresentationIndex)
    {
        NSObject *imageRepresentation = [repArray objectAtIndex:imgRepresentationIndex];
     
//		printf("Got rep %p (%zd of %d, class %s)\n", imageRepresentation, imgRepresentationIndex, repArray.count, object_getClassName(imageRepresentation));
        if ([imageRepresentation isKindOfClass:[NSBitmapImageRep class]])
		{
			CHECK(result == nil);	// If we fail this then there are two different bitmap representations stored
											// (need to decide what to do then...)
			result = (NSBitmapImageRep*)imageRepresentation;
		}
	}
	
	if (result == nil)
	{
		// It is possible to end up with a representation of type NSCGImageSnapshotRep under some circumstances
		// In that case I have to explicitly draw it into a bitmap
		[image lockFocus];
		result = [[[NSBitmapImageRep alloc] initWithFocusedViewRect:NSMakeRect(0.0, 0.0, [image size].width, [image size].height)] autorelease];
		[image unlockFocus];
	}
	
	ALWAYS_ASSERT(result != nil);		// If we fail this then there was no bitmap representation stored (vector image?)
	return [[result retain] autorelease];
}

NSBitmapImageRep *RawBitmapFromImagePath(NSString *imagePath)
{
	NSImage *theImage = [[[NSImage alloc] initWithContentsOfFile:imagePath] autorelease];
	return RawBitmapFromImage(theImage);
}

NSImage *AllocNSImageFromFile(const char *path)
{
	// Not sure if I use this for anything any more, but it was intended to allow C code
	// to load an image file into an NSImage (which the C code could treat as an opaque pointer)
	return [[NSImage alloc] initWithContentsOfFile:[SWF:@"%s", path]];
}

void ReleaseNSImage(NSImage *image)
{
	// Allows C code to release an NSImage
	[image release];
}

void BrightenNSImage(NSImage *image, int factor)
{
	// Almost certainly obsolete code, doing a quick-and-dirty brightening operation on an NSImage
	if (factor == 1)
		return;
		
	// Very dodgy way of brightening an NSImage - should come up with a proper way of doing it!
	NSRect theRect = NSMakeRect(0, 0, image.size.width, image.size.height);
	NSImage *imageCopy = [image copy];
	[image lockFocus];
	for (int i = 1; i < factor; i++)
		[imageCopy drawInRect:theRect fromRect:theRect operation:NSCompositePlusLighter fraction:1.0];
	[imageCopy release];
	[image unlockFocus];
}

NSBitmapImageRep *TintBitmap(NSBitmapImageRep *srcBitmap, NSColor *tint, NSColor *saturation, double exposureOnScreen)
{
	// Takes a greyscale input image and returns a colour result image
	// that has been tinted according to the input colours
	ALWAYS_ASSERT(srcBitmap.samplesPerPixel == 1);
	int width = srcBitmap.pixelsWide, height = srcBitmap.pixelsHigh;
	NSBitmapImageRep *destBitmap = [[NSBitmapImageRep alloc]
									initWithBitmapDataPlanes:NULL		// Bitmap allocates and releases the necessary memory for us
									pixelsWide:width
									pixelsHigh:height
									bitsPerSample:8
									samplesPerPixel:4
									hasAlpha:YES
									isPlanar:NO
									colorSpaceName:NSCalibratedRGBColorSpace
									bytesPerRow:4*width
									bitsPerPixel:0];
	// We are going to fill in the bitmap data by hand, but first we need to know what the colours are.
	// There's no simple way of querying the RGB components of an arbitrary NSColor, so we do it the
	// empirical way by seeing how they come out when drawn into the bitmap!
	[NSGraphicsContext saveGraphicsState];
	NSGraphicsContext *newContext = [NSGraphicsContext graphicsContextWithBitmapImageRep:destBitmap];
	[NSGraphicsContext setCurrentContext:newContext];
	[tint setFill];
	NSBezierPath* drawingPath = [NSBezierPath bezierPath];
	[drawingPath appendBezierPathWithRect:NSMakeRect(0, height-1, 1, 1)];
	[drawingPath fill];
	[saturation setFill];
	drawingPath = [NSBezierPath bezierPath];
	[drawingPath appendBezierPathWithRect:NSMakeRect(1, height-1, 1, 1)];
	[drawingPath fill];
	[NSGraphicsContext restoreGraphicsState];
	void *srcData = [srcBitmap bitmapData];
	unsigned char *destData = [destBitmap bitmapData];
	double tintRGB[3] = { destData[0], destData[1], destData[2] };
	unsigned char saturatedRGB[3] = { destData[4], destData[5], destData[6] };
	
	// Fill in the bitmap data
	int sizeInBytes = width * height * 4;
	double invMaxVal = 1.0 / ((1 << srcBitmap.bitsPerPixel) - 1);
	bool sixteenBit = (srcBitmap.bitsPerPixel == 16) ? true : false;
	ALWAYS_ASSERT([destBitmap bytesPerPlane] == sizeInBytes);
	for (int pos = 0; pos < width * height; pos++)
	{
		double val;
		if (sixteenBit)
			val = ((unsigned short*)srcData)[pos] * invMaxVal;
		else
			val = ((unsigned char*)srcData)[pos] * invMaxVal;
		if (val == 1.0)
		{
			destData[pos*4] = saturatedRGB[0];
			destData[pos*4+1] = saturatedRGB[1];
			destData[pos*4+2] = saturatedRGB[2];
			destData[pos*4+3] = 255;
		}
		else
		{
			val *= exposureOnScreen;
			val = MAX(val, 0.0);
			val = MIN(val, 1.0);
			destData[pos*4] = (unsigned char)(val * tintRGB[0]);
			destData[pos*4+1] = (unsigned char)(val * tintRGB[1]);
			destData[pos*4+2] = (unsigned char)(val * tintRGB[2]);
			destData[pos*4+3] = 255;
		}
	}
	
	return [destBitmap autorelease];
}

NSImage *TintImage(NSImage *srcImage, NSColor *tint, NSColor *saturation, double exposureOnScreen)
{
	NSBitmapImageRep *destBitmap = TintBitmap(RawBitmapFromImage(srcImage), tint, saturation, exposureOnScreen);
	NSImage *result = [[NSImage alloc] initWithSize:NSMakeSize(destBitmap.pixelsWide, destBitmap.pixelsHigh)];
	[result addRepresentation:destBitmap];
	return [result autorelease];
}

NSPoint ImageViewCoordToImageCoord(const NSPoint &thePoint, const NSImageView *theView)
{
	/*	Converts from a point in an NSImageView (e.g. a mouse click) to the equivalent pixel coord
		in the image that is being shown in the view.
		This makes (hopefully reasonable!) assumptions about how the NSImageView draws the NSImage.
		Note that we have to allow for whitespace due to aspect ratio mismatches	*/
	double imageAspectRatio = theView.image.size.width / theView.image.size.height;
	double scaleFactor;
	NSPoint viewOrigin;

	if (theView.bounds.size.width / theView.bounds.size.height > imageAspectRatio)
	{
		// Whitespace on left and right
		viewOrigin = NSMakePoint(([theView bounds].size.width - theView.bounds.size.height * imageAspectRatio) / 2.0,
								 0);
		scaleFactor = theView.image.size.height / theView.bounds.size.height;
	}
	else
	{
		// Whitespace above and below
		viewOrigin = NSMakePoint(0,
								 ([theView bounds].size.height - theView.bounds.size.width / imageAspectRatio) / 2.0);
		scaleFactor = theView.image.size.width / theView.bounds.size.width;
	}

	return (thePoint - viewOrigin) * scaleFactor;
}

NSPoint ImageCoordToImageViewCoord(const NSPoint &thePoint, const NSImageView *theView)
{
	/*	Converts from a pixel coord in an image to a screen coordinate in an NSImageView
		that is displaying the image.
		This makes (hopefully reasonable!) assumptions about how the NSImageView draws the NSImage.
		Note that we have to allow for whitespace due to aspect ratio mismatches	*/
	double imageAspectRatio = theView.image.size.width / theView.image.size.height;
	double scaleFactor;
	NSPoint viewOrigin;

	if (theView.bounds.size.width / theView.bounds.size.height > imageAspectRatio)
	{
		// Whitespace on left and right
		viewOrigin = NSMakePoint(([theView bounds].size.width - theView.bounds.size.height * imageAspectRatio) / 2.0,
								 0);
		scaleFactor = theView.image.size.height / theView.bounds.size.height;
	}
	else
	{
		// Whitespace above and below
		viewOrigin = NSMakePoint(0,
								 ([theView bounds].size.height - theView.bounds.size.width / imageAspectRatio) / 2.0);
		scaleFactor = theView.image.size.width / theView.bounds.size.width;
	}

	return viewOrigin + thePoint / scaleFactor;
}

NSPoint FractionalCoordWithinImageView(const NSPoint &thePoint, const NSImageView *theView)
{
	// Note that we have to allow for whitespace due to aspect ratio mismatches
	float imageAspectRatio = theView.image.size.width / theView.image.size.height, imageWidthInView;
	NSPoint viewCentre = { [theView bounds].size.width / 2,
						   [theView bounds].size.height / 2 };
	if (theView.bounds.size.width / theView.bounds.size.height > imageAspectRatio)
	{
		// Whitespace on left and right
		imageWidthInView = imageAspectRatio * theView.bounds.size.height;
	}
	else
	{
		// Whitespace above and below
		imageWidthInView = theView.bounds.size.width;
	}

	NSPoint result = (thePoint - viewCentre) / imageWidthInView;
	return result;
}

double GaussianRandom(void)
{
	static int counter = 2;
	static double randomNumbers[2];
	
	if (counter == 2)
	{
		double x1, x2, w;
		
		do {
			x1 = 2.0 * random_01() - 1.0;
			x2 = 2.0 * random_01() - 1.0;
			w = x1 * x1 + x2 * x2;
		} while ( w >= 1.0 || w == 0.0 );
		
		w = sqrt( (-2.0 * log( w ) ) / w );
		randomNumbers[0] = x1 * w;
		randomNumbers[1] = x2 * w;
		counter = 0;
	}
	counter++;
	return randomNumbers[counter - 1];
}

template<class DataType> void AddNoiseToData(DataType *data, size_t numElements, double level)
{
	int maxVal = (DataType)-1;
	for (size_t i = 0; i < numElements; i++)
	{
		double noise = GaussianRandom() * level;
		int val = int(data[i] + noise);
		data[i] = (DataType)round(LIMIT(val, 0, maxVal));
	}
}

void AddNoiseToImage(NSImage *image, double level)
{
	NSBitmapImageRep *bitmap = RawBitmapFromImage(image);
	if (bitmap.bitsPerPixel == 8)
		AddNoiseToData((unsigned char*)bitmap.bitmapData, bitmap.bytesPerPlane, level);
	else
	{
		ALWAYS_ASSERT(bitmap.bitsPerPixel == 16);
		AddNoiseToData((unsigned short*)bitmap.bitmapData, bitmap.bytesPerPlane / 2, level);
	}
}
