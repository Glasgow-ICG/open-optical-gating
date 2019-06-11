/*
 	jMovieBuilder.cpp
 
 	Copyright 2010-2015 Jonathan Taylor. All rights reserved.

	Create a movie file based on a series of image frames passed to us by the caller.
	Uses QuickTime, and is currently OS X-only (though should be possible to port
	to other operating systems on which QuickTime is available).

	Usage:
	{
		JMovieBuilder movie = JMovieBuilder(codecType, dataRef, dataRefType, boundsRect, frameRate, &err, quality);
		
		for (frameNum = 1; frameNum <= numFrames; frameNum++)
		{
			movie.AddFrame(pixelBuffer);
			// OR, on OS X:
			movie.AddFrame(myNSImage, cropRect);
		}
		// JMovieBuilder destructor does the required cleanup automatically when it goes out of scope
	}
	
*/
	

#include "jMovieBuilder.h"
#include "jUtils.h"

#if HAS_OS_X_GUI

JMovieBuilder::JMovieBuilder(OSType inCodec, Handle outputMovieDataRef, OSType outputMovieDataRefType, const BoundsRect &inBounds, double inFrameRate, OSStatus *outErr, int inQuality)
{
	// Construct a MovieBuilder object.
	// outputMovieDataRef can be created using the call QTNewDataReferenceFromCFURL, for example.
	DoInit(inCodec, inBounds, inFrameRate, outputMovieDataRef, outputMovieDataRefType, outErr, inQuality);
}

JMovieBuilder::JMovieBuilder(OSType inCodec, const char *destPath, const BoundsRect &inBounds, double inFrameRate, OSStatus *outErr, int inQuality)
{
    OSStatus err;
    Handle outputMovieDataRef = NULL;
    OSType outputMovieDataRefType;
    CFStringRef pathString = CFStringCreateWithCString(kCFAllocatorDefault, destPath, kCFStringEncodingMacRoman);
    CFURLRef outputMovieURL = CFURLCreateWithFileSystemPath(kCFAllocatorDefault, pathString, kCFURLPOSIXPathStyle, false);
    err = QTNewDataReferenceFromCFURL((CFURLRef)outputMovieURL, 0, &outputMovieDataRef, &outputMovieDataRefType);
    ALWAYS_ASSERT_NOERR(err);
    DoInit(inCodec, inBounds, inFrameRate, outputMovieDataRef, outputMovieDataRefType, outErr, inQuality);
    CFRelease(outputMovieURL);
    CFRelease(pathString);
}

void JMovieBuilder::DoInit(OSType inCodec, const BoundsRect &bounds, double frameRate, Handle outputMovieDataRef, OSType outputMovieDataRefType, OSStatus *outErr, int inQuality)
{
    pixelBufferAttributes = NULL;
	width = bounds.w;
	height = bounds.h;
	codecType = inCodec;
	desiredFramesPerSecond = frameRate;
	timeScale = (TimeScale)(desiredFramesPerSecond * 100);
	frameDuration = (TimeScale)(timeScale / desiredFramesPerSecond);
	verbose = false;
	outputVideoMedia = NULL;
	quality = inQuality;
	frameCounter = 0;
	
	// Create a new movie file. 
	OSStatus err = CreateMovieStorage(outputMovieDataRef, outputMovieDataRefType, 'TVOD', 0, 
										createMovieFileDeleteCurFile, &outputMovieDataHandler, &outputMovie);
	if (outErr == NULL)
		ALWAYS_ASSERT_NOERR(err);
	else 
	{
		*outErr = err;
		if (err != noErr)
			return;		// If we get here, then most likely is that file is open, which should be reported to the user
	}
	
	CreateCompressionSession(&compressionSession);
								
	// Make a CFDictionary that describes the pixel buffers we will use for the compression session
	// We want them to be the right dimensions and pixel format; we want them to be compatible with 
	// CGBitmapContext and CGImage.  
	pixelFormat = k32ARGBPixelFormat;
	pixelBufferAttributes = CFDictionaryCreateMutable( NULL, 0, &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks );
	ALWAYS_ASSERT(pixelBufferAttributes != NULL);
	CFNumberRef number = CFNumberCreate( NULL, kCFNumberIntType, &width );
	CFDictionaryAddValue( pixelBufferAttributes, kCVPixelBufferWidthKey, number );
	CFRelease( number );
	number = CFNumberCreate( NULL, kCFNumberIntType, &height );
	CFDictionaryAddValue( pixelBufferAttributes, kCVPixelBufferHeightKey, number );
	CFRelease( number );
	number = CFNumberCreate( NULL, kCFNumberSInt32Type, &pixelFormat );
	CFDictionaryAddValue( pixelBufferAttributes, kCVPixelBufferPixelFormatTypeKey, number );
	CFRelease( number );
	CFDictionaryAddValue( pixelBufferAttributes, kCVPixelBufferCGBitmapContextCompatibilityKey, kCFBooleanTrue );
	CFDictionaryAddValue( pixelBufferAttributes, kCVPixelBufferCGImageCompatibilityKey, kCFBooleanTrue );								
}

JMovieBuilder::~JMovieBuilder()
{
	if (frameCounter)
	{
		OSStatus result = ICMCompressionSessionCompleteFrames(compressionSession, true, 0, 0);
		ALWAYS_ASSERT_NOERR(result);
		ICMCompressionSessionRelease(compressionSession);
	}
    if (pixelBufferAttributes != NULL)
        CFRelease(pixelBufferAttributes);
	
	if (outputMovie)
		FinishOutputMovie();
}

void JMovieBuilder::CreateCompressionSession(ICMCompressionSessionRef *compressionSessionOut)
{
	// Begin a compression session that will help build the movie from the individual frames we are passed
	OSStatus err = noErr;
	ICMEncodedFrameOutputRecord encodedFrameOutputRecord;
	encodedFrameOutputRecord.encodedFrameOutputCallback = NULL;
	ICMCompressionSessionOptionsRef sessionOptions = NULL;
	
	err = ICMCompressionSessionOptionsCreate( NULL, &sessionOptions );
	ALWAYS_ASSERT_NOERR(err);
	
	// We must set these flags to enable P and B frames.
	err = ICMCompressionSessionOptionsSetAllowTemporalCompression( sessionOptions, true );
	ALWAYS_ASSERT_NOERR(err);
	err = ICMCompressionSessionOptionsSetAllowFrameReordering( sessionOptions, true );
	ALWAYS_ASSERT_NOERR(err);
	
	// Set the maximum key frame interval, also known as the key frame rate.
	err = ICMCompressionSessionOptionsSetMaxKeyFrameInterval( sessionOptions, 30 );
	ALWAYS_ASSERT_NOERR(err);

	// We need durations when we store frames.
	err = ICMCompressionSessionOptionsSetDurationsNeeded( sessionOptions, true );
	ALWAYS_ASSERT_NOERR(err);

#if 0
	// Set the average data rate.
	err = ICMCompressionSessionOptionsSetProperty( sessionOptions, 
				kQTPropertyClass_ICMCompressionSessionOptions,
				kICMCompressionSessionOptionsPropertyID_AverageDataRate,
				sizeof( averageDataRate ),
				&averageDataRate );
	ALWAYS_ASSERT_NOERR(err);
#else
	// Set the compression quality.
	err = ICMCompressionSessionOptionsSetProperty( sessionOptions, 
				kQTPropertyClass_ICMCompressionSessionOptions,
				kICMCompressionSessionOptionsPropertyID_Quality,
				sizeof( quality ),
				&quality );
	ALWAYS_ASSERT_NOERR(err);
#endif
	
	encodedFrameOutputRecord.encodedFrameOutputCallback = WriteEncodedFrameToMovie;
	encodedFrameOutputRecord.encodedFrameOutputRefCon = this;
	encodedFrameOutputRecord.frameDataAllocator = NULL;

	err = ICMCompressionSessionCreate( NULL, width, height, codecType, timeScale,
			sessionOptions, NULL, &encodedFrameOutputRecord, compressionSessionOut );
	ALWAYS_ASSERT_NOERR(err);
	
	ICMCompressionSessionOptionsRelease( sessionOptions );
}

void JMovieBuilder::CreateVideoMedia(ImageDescriptionHandle imageDesc, TimeScale timescale )
{
	// Create a video track and media to hold encoded frames.
	// This is called the first time we get an encoded frame back from the compression session.
	OSStatus err = noErr;
	Fixed trackWidth, trackHeight;
	Track outputTrack = NULL;
	
	err = ICMImageDescriptionGetProperty( 
			imageDesc,
			kQTPropertyClass_ImageDescription, 
			kICMImageDescriptionPropertyID_ClassicTrackWidth,
			sizeof( trackWidth ),
			&trackWidth,
			NULL );
	ALWAYS_ASSERT_NOERR(err);
	
	err = ICMImageDescriptionGetProperty( 
			imageDesc,
			kQTPropertyClass_ImageDescription, 
			kICMImageDescriptionPropertyID_ClassicTrackHeight,
			sizeof( trackHeight ),
			&trackHeight,
			NULL );
	ALWAYS_ASSERT_NOERR(err);
	
	if( verbose ) {
		printf( "creating %g x %g track\n", Fix2X(trackWidth), Fix2X(trackHeight) );
	}
	
	outputTrack = NewMovieTrack( outputMovie, trackWidth, trackHeight, 0 );
	ALWAYS_ASSERT_NOERR(GetMoviesError());
	
	outputVideoMedia = NewTrackMedia( outputTrack, VideoMediaType, timescale, 0, 0 );
	ALWAYS_ASSERT_NOERR(GetMoviesError());
	
	err = BeginMediaEdits( outputVideoMedia );
	ALWAYS_ASSERT_NOERR(err);
	didBeginVideoMediaEdits = true;
}

void JMovieBuilder::ReleaseBackingStorage(void *releaseRefCon, const void *baseAddress)
{
	delete[] (char *)baseAddress;
}

void JMovieBuilder::AddFrame(const CVPixelBufferRef pixelBuffer)
{
	// Feed the frame to the compression session
	OSStatus err = ICMCompressionSessionEncodeFrame(compressionSession, pixelBuffer,
										   frameCounter * frameDuration, frameDuration, kICMValidTime_DisplayTimeStampIsValid | kICMValidTime_DisplayDurationIsValid,
										   NULL, NULL, NULL);
	ALWAYS_ASSERT_NOERR(err);
	frameCounter++;
}

OSStatus JMovieBuilder::WriteEncodedFrameToMovie(void *encodedFrameOutputRefCon,
															   ICMCompressionSessionRef session, 
															   OSStatus err,
															   ICMEncodedFrameRef encodedFrame,
															   void *reserved )
{
	// This is the tracking callback function for the compression session.
	// Write the encoded frame to the movie file.
	// Note that this function adds each sample separately; better chunking can be achieved
	// by flattening the movie after it is finished, or by grouping samples, writing them in
	// groups to the data reference manually, and using AddSampleTableToMedia.
	ALWAYS_ASSERT(err == noErr);
	JMovieBuilder *us = (JMovieBuilder *)encodedFrameOutputRefCon;
	return us->WriteEncodedFrameToMovie2(session, encodedFrame);
}

OSStatus JMovieBuilder::WriteEncodedFrameToMovie2(ICMCompressionSessionRef session, 
														ICMEncodedFrameRef encodedFrame)
{
	ImageDescriptionHandle imageDesc = NULL;
	TimeValue64 decodeDuration;
	
	OSStatus err = ICMEncodedFrameGetImageDescription( encodedFrame, &imageDesc );
	ALWAYS_ASSERT_NOERR(err);
	
	if(!outputVideoMedia)
		CreateVideoMedia( imageDesc, ICMEncodedFrameGetTimeScale( encodedFrame ));
	
	decodeDuration = ICMEncodedFrameGetDecodeDuration( encodedFrame );
	if( decodeDuration == 0 ) 
	{
		// You can't add zero-duration samples to a media.  If you try you'll just get invalidDuration back.
		// Because we don't tell the ICM what the source frame durations are,
		// the ICM calculates frame durations using the gaps between timestamps.
		// It can't do that for the final frame because it doesn't know the "next timestamp"
		// (because in this example we don't pass a "final timestamp" to ICMCompressionSessionCompleteFrames).
		// So we'll give the final frame our minimum frame duration.
		decodeDuration = frameDuration * ICMEncodedFrameGetTimeScale( encodedFrame ) / timeScale;
	}
	
	if (verbose)
	{
		printf( "adding %ld byte sample: decode duration %ld, display offset %ld, flags %#lx", 
				(long)ICMEncodedFrameGetDataSize( encodedFrame ),
				(long)decodeDuration, 
				(long)ICMEncodedFrameGetDisplayOffset( encodedFrame ),
				(long)ICMEncodedFrameGetMediaSampleFlags( encodedFrame ) );
		if( true )
		{
			ICMValidTimeFlags validTimeFlags = ICMEncodedFrameGetValidTimeFlags( encodedFrame );
			if( kICMValidTime_DecodeTimeStampIsValid & validTimeFlags )
				printf( ", decode time stamp %ld", (long)ICMEncodedFrameGetDecodeTimeStamp( encodedFrame ) );
			if( kICMValidTime_DisplayTimeStampIsValid & validTimeFlags )
				printf( ", display time stamp %ld", (long)ICMEncodedFrameGetDisplayTimeStamp( encodedFrame ) );
		}
		printf( "\n" );
	}
	
	err = AddMediaSample2(
		outputVideoMedia,
		ICMEncodedFrameGetDataPtr( encodedFrame ),
		ICMEncodedFrameGetDataSize( encodedFrame ),
		decodeDuration,
		ICMEncodedFrameGetDisplayOffset( encodedFrame ),
		(SampleDescriptionHandle)imageDesc,
		1,
		ICMEncodedFrameGetMediaSampleFlags( encodedFrame ),
		NULL );
	ALWAYS_ASSERT_NOERR(err);
	
	// Note: if you don't need to intercept any values, you could equivalently call:
	// err = AddMediaSampleFromEncodedFrame( outputVideoMedia, encodedFrame, NULL );
	// if( err ) {
	//     fprintf( stderr, "AddMediaSampleFromEncodedFrame() failed (%d)\n", (int)err );
	//     goto bail;
	// }

	return noErr;
}


void JMovieBuilder::FinishOutputMovie(void)
{
	// Clean up and complete generation of the movie file
	OSStatus err = noErr;
	Track videoTrack = NULL;
	
	if (didBeginVideoMediaEdits)
	{
		// End the media sample-adding session.
		err = EndMediaEdits(outputVideoMedia);
		ALWAYS_ASSERT_NOERR(err);
	}	
	
	// Make sure things are extra neat.
	ExtendMediaDecodeDurationToDisplayEndTime(outputVideoMedia, NULL );
	
	// Insert the stuff we added into the track, at the end.
	videoTrack = GetMediaTrack(outputVideoMedia);
	err = InsertMediaIntoTrack(videoTrack, 
			GetTrackDuration(videoTrack), 
			0, GetMediaDisplayDuration(outputVideoMedia), // NOTE: use this instead of GetMediaDuration
			fixed1);
	ALWAYS_ASSERT_NOERR(err);
	
	// Write the movie header to the file.
	err = AddMovieToStorage(outputMovie, outputMovieDataHandler);
	ALWAYS_ASSERT_NOERR(err);
	
	CloseMovieStorage(outputMovieDataHandler);
	outputMovieDataHandler = 0;
	
	DisposeMovie(outputMovie);
}

MoviePixelBuffer::MoviePixelBuffer(const BoundsRect &bounds)
{
	// config
	int flag = true;
	yes = CFNumberCreate(NULL, kCFNumberIntType, &flag );
	d = CFDictionaryCreateMutable(NULL, 2, NULL, NULL);
	CFDictionaryAddValue(d, kCVPixelBufferCGImageCompatibilityKey, yes);
	CFDictionaryAddValue(d, kCVPixelBufferCGBitmapContextCompatibilityKey, yes);
		
	// create pixel buffer
	buffer = NULL;
	CVPixelBufferCreate(kCFAllocatorDefault, bounds.w, bounds.h, k32ARGBPixelFormat, (CFDictionaryRef)d, &buffer);
	CVPixelBufferLockBaseAddress(buffer, 0);
	baseAddr = (unsigned char *)CVPixelBufferGetBaseAddress(buffer);
	rowBytes = CVPixelBufferGetBytesPerRow(buffer);
}

MoviePixelBuffer::~MoviePixelBuffer()
{
	CVPixelBufferUnlockBaseAddress(buffer, 0);
	CVPixelBufferRelease(buffer);
	CFRelease(d);
	CFRelease(yes);
}

#endif
