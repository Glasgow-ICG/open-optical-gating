/*	Module: JMovieBuilder.cpp

	Create a movie file based on a simulation replay. Uses QuickTime, and is currently
	OS X-only (though should be possible to port to other operating systems on which
	QuickTime is available.

	Usage:

	{
		JMovieBuilder movie = (movieType, codecType, moviePath, &boundsRect, frameRate);
		
		for (frameNum = 1; frameNum <= numFrames; frameNum++)
		{
			movie.BeginMovieFrame();
			// (Insert code here to draw the frame)
			movie->EndMovieFrame();
		}
		// JMovieBuilder destructor does the required cleanup automatically when it goes out of scope
	}
	
	n.b. some of the AVI code based on freely available code written by Mark Asbach, Institute of Communications Engineering, RWTH Aachen University
	n.b. JBetterMovieBuilder is very heavily based on the Apple sample code CaptureAndCompressIPBMovie.c

*/
	

#include "jMovieBuilder.h"
#include "jUtils.h"
#include <algorithm>

#if OS_X

#define		kPixelDepth 		32

static StringPtr ConvertCToPascalString2 (const char *theString, Str255 pStr)
{
	snprintf((char *)pStr + 1, 255, "%s", theString);
	pStr[0] = MIN((size_t)255, strlen(theString));
	return(pStr);
}

BaseMovieBuilder::BaseMovieBuilder(const Rect *bounds)
{
	// Create a graphics world to draw a given frame into
	OSStatus err = NewGWorld (&theGWorld,		/* pointer to created gworld */	
					 kPixelDepth,		/* pixel depth */
					 bounds, 		/* bounds */
					 nil, 				/* color table */
					 nil,				/* handle to GDevice */ 
					 (GWorldFlags)0);	/* flags */
	ALWAYS_ASSERT_NOERR(err);

	// Lock the pixels
	LockPixels(GetPortPixMap(theGWorld));
}

void BaseMovieBuilder::GetDestinationDetails(const char *inFileName, Handle *outputMovieDataRef, OSType *outputMovieDataRefType)
{
	// Note that there is a partner function GetDestinationDetailsUsingSheetOnWindow that implements a modern sheet-based version of this
	OSStatus err = noErr;
	*outputMovieDataRef = NULL;
	*outputMovieDataRefType = 0;
	
	if (inFileName == NULL)
	{
		// Prompt the user for an output file.
		NavDialogCreationOptions navOptions;
		navOptions.version = kNavDialogCreationOptionsVersion;
		navOptions.optionFlags = 0;
		NavDialogRef navDialog = NULL;
		NavReplyRecord navReply;
		navReply.version = kNavReplyRecordVersion;
		AEDesc actualDesc = { 0, 0 };
		FSRef parentFSRef;
		
		err = NavGetDefaultDialogCreationOptions( &navOptions );
		ALWAYS_ASSERT_NOERR(err);
		navOptions.windowTitle = CFSTR("Save Captured Movie As...");
		navOptions.message = CFSTR("Pick where to save the captured and compressed movie.");
		navOptions.saveFileName = CFSTR("captured.mov");
		navOptions.modality = kWindowModalityAppModal;
		
		err = NavCreatePutFileDialog( &navOptions, MovieFileType, 'TVOD', NULL, NULL, &navDialog );
		ALWAYS_ASSERT_NOERR(err);
		
		err = NavDialogRun( navDialog );
		ALWAYS_ASSERT_NOERR(err);
		
		if (NavDialogGetUserAction( navDialog ) != kNavUserActionSaveAs)
			return;	// With null dataRef
		
		err = NavDialogGetReply( navDialog, &navReply );
		ALWAYS_ASSERT_NOERR(err);
		
		err = AECoerceDesc( &navReply.selection, typeFSRef, &actualDesc );
		ALWAYS_ASSERT_NOERR(err);

		err = AEGetDescData( &actualDesc, &parentFSRef, sizeof( FSRef ) );
		ALWAYS_ASSERT_NOERR(err);
		
		err = QTNewDataReferenceFromFSRefCFString( &parentFSRef, navReply.saveFileName, 0, outputMovieDataRef, outputMovieDataRefType );
		ALWAYS_ASSERT_NOERR(err);
		
		NavDisposeReply( &navReply );
		NavDialogDispose( navDialog );
		AEDisposeDesc( &actualDesc );
	}
	else
	{
		FSSpec fileSpec;
		Str255 fileName;
		err = FSMakeFSSpec(0, 0, ConvertCToPascalString2(inFileName, fileName), &fileSpec);
		if ((err != noErr) && (err != fnfErr))
		{
			fprintf(stderr, "FSMakeFSSpec: error %d with filename %.100s\n", (int)err, inFileName);
			ALWAYS_ASSERT_NOERR(err);
		}
		err =	QTNewDataReferenceFromFSSpec(&fileSpec, 0, outputMovieDataRef, outputMovieDataRefType);
		ALWAYS_ASSERT_NOERR(err);
	}
}

Rect BaseMovieBuilder::BeginMovieFrame(GWorldPtr *destPort)
{
	// Change the current graphics port to the GWorld
	GetGWorld(&oldPort, &oldGDeviceH);
	SetGWorld(theGWorld, nil);
	Rect bounds;
	GetPortBounds(theGWorld, &bounds);
	if (destPort != NULL)
		*destPort = theGWorld;
		
	return bounds;
}

JMovieBuilder::JMovieBuilder(OSType inMovieType, OSType inCodec, const char *inFileName, const Rect *bounds, float frameRate) : BaseMovieBuilder(bounds)
{
	EnterMovies();

	/*	Supported movie types:
			kQTFileTypeMovie, kQTFileTypeMP4, kQTFileTypeAVI (not many codecs supported for AVI)
		Supported compression types (codecs) are defined by the container type.
			kQTFileTypeMovie can contain just about any compression type
			kQTFileTypeMP4 will always compress in MP4 regardless of the codec specified
			kQTFileTypeAVI only support a very small number of codecs (which as far as I can tell are kRawCodecType, kCinepakCodecType, kBMPCodecType, kDVCNTSCCodecType and other DV types)
	*/
	if (inMovieType == 0)
		movieType = kQTFileTypeMovie;
	else
		movieType = inMovieType;
	if (inCodec == 0)
		compressionTypeToUse = kH264CodecType;
	else
		compressionTypeToUse = inCodec;
	if ((movieType == kQTFileTypeMP4) && (compressionTypeToUse != kMPEG4VisualCodecType))
		printf("WARNING: MP4 file containers only support MP4 codec - that will be used automatically\n");

	// The exporter needs to run on a separate thread if this is going to fit in with the BaseMovieBuilder API
	lastTrackWritten = -1;
	currentTrackReady = -1;
	frameToGenerate = -1;
	snprintf(cFilename, sizeof(cFilename), "%s", inFileName);
	
	ComponentDescription cd; 
	cd.componentType          = MovieExportType; 
	cd.componentSubType       = movieType;
	cd.componentManufacturer  = 0;                //'appl'; 
	cd.componentFlags         = canMovieExportFromProcedures; 
	cd.componentFlagsMask     = canMovieExportFromProcedures; 
	
	// find component
	Component             fcomponent;
	fcomponent = FindNextComponent (nil, &cd);
	ALWAYS_ASSERT(fcomponent);
	ci = OpenComponent (fcomponent); 
	ALWAYS_ASSERT(ci);

	Handle name = NewHandle(0);
	OSErr e = GetComponentInfo(fcomponent, NULL, name, NULL, NULL);
	ALWAYS_ASSERT_NOERR(e);
	char *cName = *name;
	int len = *cName++;
	cName[len] = 0;
//	printf("Component name %s\n", cName);

//	Boolean canceled;
//	ComponentResult res = MovieExportDoUserDialog(ci, NULL, NULL, 0, 0, &canceled);
//	ALWAYS_ASSERT_NOERR(res);

	
	PixMapHandle pmap = GetGWorldPixMap(theGWorld);
	LockPixels(pmap); 
	e = MakeImageDescriptionForPixMap(pmap, &imageDescription);
    ALWAYS_ASSERT_NOERR(e);

	LocalGetMutex lgm(&commsMutex);

	int result = pthread_cond_init(&dataReadySignal, NULL);
	ALWAYS_ASSERT(result == 0);

	result = pthread_create(&workerThread, NULL, WorkThreadCallback, this);
	ALWAYS_ASSERT(result == 0);
	
	// Wait for worker thread to be ready for the first frame
	while (frameToGenerate != 0)
	{
		printf("block waiting for dataReadySignal\n");
		commsMutex.BlockWaitingForSignal(&dataReadySignal, __LINE__);
	}
}

JMovieBuilder::~JMovieBuilder()
{
	currentTrackReady = -2;		// Indicates end of file
	{
		LocalGetMutex lgm(&commsMutex);
		pthread_cond_signal(&dataReadySignal);
	}
	// Wait for worker thread to terminate
	pthread_join(workerThread, NULL);

	int result = pthread_cond_destroy(&dataReadySignal);
	ALWAYS_ASSERT(result == 0);
}

void *JMovieBuilder::WorkThreadCallback(void *ref)
{
	((JMovieBuilder *)ref)->WorkThread();
	return NULL;
}

void JMovieBuilder::EndMovieFrame(void)
{
	// Indicate that there is a frame ready
	LocalGetMutex lgm(&commsMutex);
	currentTrackReady++;
	pthread_cond_signal(&dataReadySignal);
	
	// Now wait for the frame to be written
	while (lastTrackWritten < currentTrackReady)
	{
		commsMutex.BlockWaitingForSignal(&dataReadySignal, __LINE__);
	}
}

void JMovieBuilder::WorkThread(void)
{
	ComponentResult  res;
	printf("WorkThread()\n");
	// configure component
	#define kMySampleRate      2997      // 29.97 fps 
	#define kMyFrameDuration    100      // one frame at 29.97 fps
	res = MovieExportAddDataSource (ci, VideoMediaType, kMySampleRate, 
	                                &trackID, 
									NewMovieExportGetPropertyUPP (getVideoPropertyProc), 
	                                NewMovieExportGetDataUPP     (getVideoDataProc), 
									this); 
	ALWAYS_ASSERT_NOERR(res);

	// create file alias
	FSSpec fileSpec = { 0, 0, "\p" };
	ConvertCToPascalString2(cFilename, fileSpec.name);
	
	myDataRef = NewHandle (sizeof (Handle));
	OSErr err = QTNewAlias (&fileSpec, (AliasHandle *)&myDataRef, 0);
	if ( err != noErr && err != fnfErr )
	{
		SysError(err);
		ALWAYS_ASSERT(0);
	}
	
	// remove any existing movie file (otherwise we will just append...)
	// Might be better to do it in a way that keeps the old file if the export fails for some reason.
	// This will do for now, though.
	err =  DeleteMovieStorage(myDataRef, rAliasType);
	if (err != fnfErr)
		ASSERT_NOERR(err);

	// start output
	err = MovieExportFromProceduresToDataRef(ci, myDataRef, rAliasType); 
	ASSERT_NOERR(err);
}

pascal OSErr JMovieBuilder::getVideoPropertyProc(void * refcon, long trackID, OSType propertyType, void * propertyValue)
{
	return ((JMovieBuilder *)refcon)->GetVideoProperty(trackID, propertyType, propertyValue);
}

OSErr JMovieBuilder::GetVideoProperty(long callbackTrackID, OSType propertyType, void * propertyValue)
{
	ALWAYS_ASSERT(callbackTrackID == trackID);		// Sanity check
	
    OSErr err = noErr;
	
	Rect gWorldRect;
	GetPortBounds(theGWorld, &gWorldRect);
	
    switch (propertyType) 
	{
        case movieExportWidth:
            *(Fixed *) propertyValue = (gWorldRect.right - gWorldRect.left) << 16; 
            break; 
        case movieExportHeight: 
            *(Fixed *) propertyValue = (gWorldRect.bottom - gWorldRect.top) << 16; 
            break; 
        case scSpatialSettingsType: 
			{
				SCSpatialSettings *ss = (SCSpatialSettings *)propertyValue; 
				ss->codecType      = compressionTypeToUse;
				ss->codec          = 0;
				ss->depth          = 0;
				ss->spatialQuality = codecHighQuality;
            }
            break; 
        default: 
            err = paramErr; 
            break;
    }
    return err; 
}


pascal OSErr JMovieBuilder::getVideoDataProc (void * refCon, MovieExportGetDataParams * params)
{
	return ((JMovieBuilder *)refCon)->GetVideoData(params);
}

OSErr JMovieBuilder::GetVideoData(MovieExportGetDataParams * params)
{ 
    OSErr               err   = noErr; 
    
	LocalGetMutex lgm(&commsMutex);

	// Signal that we have finished working with the previous frame (if there was one)
	frameToGenerate = lastTrackWritten + 1;
	printf("dataReadySignal (ftg = %ld)\n", frameToGenerate);
	pthread_cond_signal(&dataReadySignal);

	// Wait for the next frame to be ready
	while (lastTrackWritten == currentTrackReady)
		commsMutex.BlockWaitingForSignal(&dataReadySignal, __LINE__);

	// Terminate if required
	if (currentTrackReady == -2) 
        return eofErr;
		
	// writing data
	params->dataPtr           = GetPixBaseAddr(GetGWorldPixMap(theGWorld)); 
    params->dataSize          = (**imageDescription).dataSize; 
    params->actualTime        = params->requestedTime; 
    params->descType          = VideoMediaType; 
    params->descSeed          = 1; 
    params->desc              = (SampleDescriptionHandle) imageDescription; 
    params->durationPerSample = kMyFrameDuration; 
    params->sampleFlags       = 0; 

	lastTrackWritten = currentTrackReady;

	return err;
}

JBetterMovieBuilder::JBetterMovieBuilder(OSType inCodec, const char *inFileName, const Rect *inBounds, float inFrameRate, OSStatus *outErr, int inQuality) : BaseMovieBuilder(inBounds)
{
	Handle outputMovieDataRef;
	OSType outputMovieDataRefType;
	GetDestinationDetails(inFileName, &outputMovieDataRef, &outputMovieDataRefType);
	DoInit(inCodec, inBounds, inFrameRate, outputMovieDataRef, outputMovieDataRefType, outErr, inQuality);
	DisposeHandle(outputMovieDataRef);
}

JBetterMovieBuilder::JBetterMovieBuilder(OSType inCodec, Handle outputMovieDataRef, OSType outputMovieDataRefType, const Rect *inBounds, float inFrameRate, OSStatus *outErr, int inQuality) : BaseMovieBuilder(inBounds)
{
	DoInit(inCodec, inBounds, inFrameRate, outputMovieDataRef, outputMovieDataRefType, outErr, inQuality);
}

void JBetterMovieBuilder::DoInit(OSType inCodec, const Rect *bounds, float frameRate, Handle outputMovieDataRef, OSType outputMovieDataRefType, OSStatus *outErr, int inQuality)
{
	width = bounds->right - bounds->left;
	height = bounds->bottom - bounds->top;
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
			return;		// Most likely is that file is open, which should be reported to the user
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

JBetterMovieBuilder::~JBetterMovieBuilder()
{
	if (frameCounter)
	{
		OSStatus result = ICMCompressionSessionCompleteFrames(compressionSession, true, 0, 0);
		ALWAYS_ASSERT_NOERR(result);
		ICMCompressionSessionRelease(compressionSession);
	}
	CFRelease(pixelBufferAttributes);
	
	if (outputMovie)
		FinishOutputMovie();
}

void JBetterMovieBuilder::CreateCompressionSession(ICMCompressionSessionRef *compressionSessionOut)
{
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
	err = ICMCompressionSessionOptionsSetMaxKeyFrameInterval( sessionOptions, 10 );
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

// Create a video track and media to hold encoded frames.
// This is called the first time we get an encoded frame back from the compression session.
void JBetterMovieBuilder::CreateVideoMedia( 
							ImageDescriptionHandle imageDesc,
							TimeScale timescale )
{
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

void JBetterMovieBuilder::ReleaseBackingStorage(void *releaseRefCon, const void *baseAddress)
{
	delete[] (char *)baseAddress;
}

void JBetterMovieBuilder::EndMovieFrame(void)
{
	// Unfortunately the ICM code wants to passed a CVPixelBuffer.
	// I suppose we'll have to copy our data into that...
	// On the plus side this would make it very easy to also support CGContext drawing for movie frame drawing
	// (see CaptureAndCompressIPBMovie sample code for how to shoehorn the CVPixelBuffer into a CGContext...)
	// We need to provide backing storage because the PixMap will soon have the next frame rendered into it.
	CVPixelBufferRef pixelBuffer;
	PixMapHandle thePixMap = GetPortPixMap(theGWorld);
	size_t pixMapSizeInBytes = height * GetPixRowBytes(thePixMap);
	char *backingStorage = new char[pixMapSizeInBytes];
	memcpy(backingStorage, GetPixBaseAddr(thePixMap), pixMapSizeInBytes);
	OSStatus err = CVPixelBufferCreateWithBytes(CFAllocatorGetDefault(),
												width, height,
												pixelFormat,
												backingStorage,
												GetPixRowBytes(thePixMap),
												ReleaseBackingStorage,
												NULL,
												pixelBufferAttributes,
												&pixelBuffer);
	ALWAYS_ASSERT_NOERR(err);
						
	// Feed the frame to the compression session and then release the CVBuffer
	err = ICMCompressionSessionEncodeFrame(compressionSession, pixelBuffer,
					frameCounter * frameDuration, frameDuration, kICMValidTime_DisplayTimeStampIsValid | kICMValidTime_DisplayDurationIsValid,
					NULL, NULL, NULL);
	ALWAYS_ASSERT_NOERR(err);
	CVPixelBufferRelease(pixelBuffer);
	frameCounter++;

	SetGWorld (oldPort, oldGDeviceH);
}

void JBetterMovieBuilder::AddFrame(const NSImage *frameImage, const _NSRect *cropRect, double gain)
{
	BeginMovieFrame();
	void CopyNSImageToGWorld(const NSImage *image, GWorldPtr gWorldPtr, const _NSRect *cropRect, double gain);		// Prototyped here because this is in a cocoa file, whose header I don't want to include from this C++ code
	CopyNSImageToGWorld(frameImage, theGWorld, cropRect, gain);
	EndMovieFrame();
}

// This is the tracking callback function for the compression session.
// Write the encoded frame to the movie file.
// Note that this function adds each sample separately; better chunking can be achieved
// by flattening the movie after it is finished, or by grouping samples, writing them in 
// groups to the data reference manually, and using AddSampleTableToMedia.
OSStatus JBetterMovieBuilder::WriteEncodedFrameToMovie(void *encodedFrameOutputRefCon, 
															   ICMCompressionSessionRef session, 
															   OSStatus err,
															   ICMEncodedFrameRef encodedFrame,
															   void *reserved )
{
	ALWAYS_ASSERT(err == noErr);
	JBetterMovieBuilder *us = (JBetterMovieBuilder *)encodedFrameOutputRefCon;
	return us->WriteEncodedFrameToMovie2(session, encodedFrame);
}

OSStatus JBetterMovieBuilder::WriteEncodedFrameToMovie2(ICMCompressionSessionRef session, 
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


void JBetterMovieBuilder::FinishOutputMovie(void)
{
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

#endif
