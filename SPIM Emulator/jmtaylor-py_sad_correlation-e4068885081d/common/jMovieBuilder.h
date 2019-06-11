/*
	jMovieBuilder.h
 
	Copyright 2010-2015 Jonathan Taylor. All rights reserved.

	Build up a movie file from individual frames that are supplied to this class by the caller in sequence.
*/

#ifndef __JMOVIEBUILDER_H__
#define __JMOVIEBUILDER_H__

#include "jMutex.h"
#include "BoundsRect.h"

#if (!HAS_OS_X_GUI) || defined(__x86_64__)

// Code not implemented except on 32-bit OS X
// This first bit of code here is just dummy code to allow things to compile on other platforms, but without any functionality
class JMovieBuilder
{
  public:
			JMovieBuilder(OSType inCodec, char** outputMovieDataRef, OSType outputMovieDataRefType, const BoundsRect &bounds, double frameRate, int32_t *outErr, int inQuality = 0) { ALWAYS_ASSERT(0); }
	void	AddFrame(const NSImage *frameImage, const NSRect *cropRect, double gain = 1.0) { }
};

#else

#include <Quicktime/QuickTime.h>

class JMovieBuilder
{
  protected:
	int							width;		// dest width
	int							height;		// dest height
	CodecType					codecType;	// codec
	int							quality;
	ICMCompressionSessionRef	compressionSession; // compresses frames
	Movie						outputMovie; // movie file for storing compressed frames
	Media						outputVideoMedia; // media for our video track in the movie
	DataHandler					outputMovieDataHandler; // storage for movie header
	Boolean						didBeginVideoMediaEdits;
	Boolean						verbose;
    TimeScale					timeScale;
	double						desiredFramesPerSecond;
	TimeValue					frameDuration;
	CFMutableDictionaryRef		pixelBufferAttributes;
	OSType						pixelFormat;
	int							frameCounter;
	
	void	DoInit(OSType inCodec, const BoundsRect &bounds, double frameRate, Handle outputMovieDataRef, OSType outputMovieDataRefType, OSStatus *outErr, int inQuality);
	void	SetUpOutputMovie(const char *inFileName);
	void	CreateCompressionSession(ICMCompressionSessionRef *compressionSessionOut);
	void	CreateVideoMedia(ImageDescriptionHandle imageDesc, TimeScale timescale );
	static OSStatus WriteEncodedFrameToMovie(void *encodedFrameOutputRefCon, 
											   ICMCompressionSessionRef session, 
											   OSStatus err,
											   ICMEncodedFrameRef encodedFrame,
											   void *reserved );
	OSStatus WriteEncodedFrameToMovie2(ICMCompressionSessionRef session, ICMEncodedFrameRef encodedFrame);
	static void ReleaseBackingStorage(void *releaseRefCon, const void *baseAddress);
	void	FinishOutputMovie(void);
	
  public:
			JMovieBuilder(OSType inCodec, Handle outputMovieDataRef, OSType outputMovieDataRefType, const BoundsRect &bounds, double frameRate, OSStatus *outErr, int inQuality = codecLosslessQuality);
            JMovieBuilder(OSType inCodec, const char *destPath, const BoundsRect &bounds, double frameRate, OSStatus *outErr, int inQuality = codecLosslessQuality);
	virtual ~JMovieBuilder();
	
	int Width(void) { return width; }
	int Height(void) { return height; }
	
/*	I am still working out what switch to use on this next compile-time condition.
	Originally had something like "struct _NSImage" as a placeholder for non-ObjC code, but that doesn't work on Mountain Lion.
	#ifdef __COREFOUNDATION__ doesn't seem to work (defined in some of my apparently c-only files)
	Haven't yet found a specific option that identifies files where NSImage is defined
	Trying just objc switch. I don't ~think~ I've got any C++ code that uses AddFrame...	*/
#ifdef __OBJC__
	void	AddFrame(const NSImage *frameImage, const NSRect *cropRect, double gain = 1.0);
#endif
	void	AddFrame(const CVPixelBufferRef pixelBuffer);
};

#endif

#ifdef __OBJC__
void GetDestinationDetailsUsingSheetOnWindow(NSWindow *sheetOnWindow, void (^handler)(NSInteger result, NSSavePanel *savePanel));
#endif

#if HAS_OS_X_GUI

#include <CoreVideo/CoreVideo.h>
#include "BoundsRect.h"

class MoviePixelBuffer
{
protected:
	CFNumberRef yes;
	CFMutableDictionaryRef d;
public:
	CVPixelBufferRef buffer;
	unsigned char *baseAddr;
	size_t rowBytes;
	
	MoviePixelBuffer(const BoundsRect &inBounds);
	virtual ~MoviePixelBuffer();
};
#endif

#endif
