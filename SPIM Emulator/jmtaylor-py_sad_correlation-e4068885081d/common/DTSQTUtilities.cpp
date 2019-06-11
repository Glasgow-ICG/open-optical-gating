/*
	File:		DTSQTUtilities.c

	Contains:	QuickTime functions.

	Written by: 	

	Copyright:	Copyright © 1991-2001 by Apple Computer, Inc., All Rights Reserved.

	Disclaimer:	IMPORTANT:  This Apple software is supplied to you by Apple Computer, Inc.
				("Apple") in consideration of your agreement to the following terms, and your
				use, installation, modification or redistribution of this Apple software
				constitutes acceptance of these terms.  If you do not agree with these terms,
				please do not use, install, modify or redistribute this Apple software.

				In consideration of your agreement to abide by the following terms, and subject
				to these terms, Apple grants you a personal, non-exclusive license, under AppleÕs
				copyrights in this original Apple software (the "Apple Software"), to use,
				reproduce, modify and redistribute the Apple Software, with or without
				modifications, in source and/or binary forms; provided that if you redistribute
				the Apple Software in its entirety and without modifications, you must retain
				this notice and the following text and disclaimers in all such redistributions of
				the Apple Software.  Neither the name, trademarks, service marks or logos of
				Apple Computer, Inc. may be used to endorse or promote products derived from the
				Apple Software without specific prior written permission from Apple.  Except as
				expressly stated in this notice, no other rights or licenses, express or implied,
				are granted by Apple herein, including but not limited to any patent rights that
				may be infringed by your derivative works or by other works in which the Apple
				Software may be incorporated.

				The Apple Software is provided by Apple on an "AS IS" basis.  APPLE MAKES NO
				WARRANTIES, EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION THE IMPLIED
				WARRANTIES OF NON-INFRINGEMENT, MERCHANTABILITY AND FITNESS FOR A PARTICULAR
				PURPOSE, REGARDING THE APPLE SOFTWARE OR ITS USE AND OPERATION ALONE OR IN
				COMBINATION WITH YOUR PRODUCTS.

				IN NO EVENT SHALL APPLE BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL OR
				CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
				GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
				ARISING IN ANY WAY OUT OF THE USE, REPRODUCTION, MODIFICATION AND/OR DISTRIBUTION
				OF THE APPLE SOFTWARE, HOWEVER CAUSED AND WHETHER UNDER THEORY OF CONTRACT, TORT
				(INCLUDING NEGLIGENCE), STRICT LIABILITY OR OTHERWISE, EVEN IF APPLE HAS BEEN
				ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
                
	Change History (most recent first):
				11/7/2001	srk				Carbonized
				7/28/1999	Karl Groethe	Updated for Metrowerks Codewarror Pro 2.1
				

*/


// INCLUDES
#include "DTSQTUtilities.h"
#include "path2fss.h"
#include "GetFile.h"

// MOVIE TOOLBOX FUNCTIONS

/*______________________________________________________________________
	QTUSimpleGetMovie - Get a Movie from a specific file (simpler version)

pascal OSErr QTUSimpleGetMovie(Movie *theMovie)

theMovie				will contain the selected movie when function exits.

DESCRIPTION
	QTUSimpleGetMovie is a simplified version of getting a movie from a file, no need for
	returning refnums, res IDs of keeping track of FSSpecs (compared with QTUGetMovie)
*/

#define	kTypeListCount	2

OSErr QTUSimpleGetMovie(Movie *theMovie, char filename[256])
{
	OSErr 			anErr = noErr;
	short			resFile = 0;
	short			resID = 0;
	Str255			movieName;
	Boolean			wasChanged;
//	OSType 			myTypeList[kTypeListCount] = {kQTFileTypeMovie, 'VooM'/*, kQTFileTypeQuickTimeImage*/};
	FSSpec			theFSSpec;
    
//    anErr = GetOneFileWithPreview (kTypeListCount, (TypeListPtr)&myTypeList, &theFSSpec, NULL);
    anErr = GetOneFileWithPreview (0, (TypeListPtr)NULL, &theFSSpec, NULL);
	if(anErr == noErr)
	{
		anErr = OpenMovieFile(&theFSSpec, &resFile, fsRdPerm); ASSERT_NOERR(anErr);
		if(anErr == noErr)
		{
			anErr = NewMovieFromFile(theMovie, resFile, &resID, movieName, newMovieActive, &wasChanged);
			ASSERT_NOERR(anErr);

			CloseMovieFile(resFile);
		}
	}
	
#if 1
	FSRef fsRef;
	OSStatus err = FSpMakeFSRef(&theFSSpec, &fsRef);
	ALWAYS_ASSERT(err == noErr);
	HFSUniStr255 name;
	err = FSGetCatalogInfo (&fsRef, 0, NULL, &name, NULL, NULL);
	ALWAYS_ASSERT(err == noErr);
	CFStringRef cfname = CFStringCreateWithCharacters(kCFAllocatorDefault, name.unicode, name.length);

	Boolean ok = CFStringGetCString(cfname, filename, 256, kCFStringEncodingMacRoman);
	if (!ok)
		strcpy(filename, "??");
#else
	strncpy(filename, (char *)theFSSpec.name + 1, theFSSpec.name[0]);
	filename[theFSSpec.name[0]] = 0;
#endif
	
	return anErr;
}

OSErr QTUSimpleGetMovieFromFSSpec(Movie *theMovie, char filename[256], const FSSpec *theFSSpec)
{
	OSErr 			anErr = noErr;
	short			resFile = 0;
	short			resID = 0;
	Str255			movieName;
	Boolean			wasChanged;

	anErr = OpenMovieFile(theFSSpec, &resFile, fsRdPerm);
	ASSERT_NOERR(anErr);
	anErr = NewMovieFromFile(theMovie, resFile, &resID, movieName, newMovieActive, &wasChanged);
	ASSERT_NOERR(anErr);
	CloseMovieFile(resFile);
	
	// This conversion is to avoid the truncation that occurs with long filenames in FSSpecs
	ASSERT_NOERR(anErr);
	anErr = OpenMovieFile(theFSSpec, &resFile, fsRdPerm);
	ASSERT_NOERR(anErr);
	anErr = NewMovieFromFile(theMovie, resFile, &resID, movieName, newMovieActive, &wasChanged);
	ASSERT_NOERR(anErr);
	CloseMovieFile(resFile);
	
#if 1
	FSRef fsRef;
	OSStatus err = FSpMakeFSRef(theFSSpec, &fsRef);
	ALWAYS_ASSERT(err == noErr);
	HFSUniStr255 name;
	err = FSGetCatalogInfo (&fsRef, 0, NULL, &name, NULL, NULL);
	ALWAYS_ASSERT(err == noErr);
	CFStringRef cfname = CFStringCreateWithCharacters(kCFAllocatorDefault, name.unicode, name.length);

	Boolean ok = CFStringGetCString(cfname, filename, 256, kCFStringEncodingMacRoman);
	if (!ok)
		strcpy(filename, "??");
#else
	strncpy(filename, (char *)theFSSpec->name + 1, theFSSpec->name[0]);
	filename[theFSSpec->name[0]] = 0;
#endif
	
	return anErr;
}

OSErr QTUSimpleGetMovieFromPath(Movie *theMovie, char filename[256], const char *path)
{
	OSErr 			anErr = noErr;
	FSSpec			theFSSpec;

	anErr = __path2fss(path, &theFSSpec);
	ASSERT_NOERR(anErr);
	return QTUSimpleGetMovieFromFSSpec(theMovie, filename, &theFSSpec);
}

pascal TimeValue  QTUGetDurationOfFirstMovieSample(Movie theMovie, OSType theMediaType)
{
	OSErr 			anErr = noErr;
	TimeValue		interestingDuration = 0;
	short			timeFlags = nextTimeMediaSample+nextTimeEdgeOK;

	GetMovieNextInterestingTime(theMovie, timeFlags, (TimeValue)1, &theMediaType, 0, 
													fixed1, NULL, &interestingDuration);
	anErr = GetMoviesError(); ASSERT_NOERR(anErr);

	return interestingDuration;
}

/*______________________________________________________________________
	QTUGetMovieFrameCount - Return the amount of frames in the movie based on frame rate estimate.

pascal long QTUGetMovieFrameCount(Movie theMovie, long theFrameRate)

theMovie					the movie we want to calculate the frame count for.			
theFrameRate			the expected frame rate of the movie

DESCRIPTION
	QTUGetMovieFrameCount is a simple operation that takes into account the duration of the movie,
	the time scale and a suggested frame rate, and based on this will calculate the 
	amount of frames needed in the movie. We assume that the frame rate will be uniform in the movie.
*/

pascal long QTUGetMovieFrameCount(Movie theMovie, long theFrameRate)
{
	long 		frameCount, duration, timescale;
	float 	exactFrames;
	
	ALWAYS_ASSERT(theMovie != NULL); if(theMovie == NULL) return invalidMovie;

	duration	 	= GetMovieDuration(theMovie);
	timescale 		= GetMovieTimeScale(theMovie);
	exactFrames	= (float)duration * theFrameRate;
	
	frameCount	= exactFrames / timescale / 65536;
	
	if(frameCount == 0)
		frameCount = 1;			// we got to have at least one frame
	
	return frameCount;
}

pascal OSErr QTUGetStartPointOfFirstVideoSample(Movie theMovie, TimeValue *startPoint) 
{
	*startPoint = -1;

	MoviesTask( theMovie, 0 );
	
	GetMovieNextInterestingTime(theMovie, nextTimeStep | nextTimeEdgeOK, 0, NULL, 0, 
													fixed1, startPoint, NULL);
	OSErr anErr = GetMoviesError();
	ASSERT_NOERR(anErr);

	return anErr;
}

#define   kCharacteristicHasVideoFrameRate  FOUR_CHAR_CODE('vfrr')
#define   kCharacteristicIsAnMpegTrack     FOUR_CHAR_CODE('mpeg')

void MovieGetVideoMediaAndMediaHandler(Movie inMovie, Media *outMedia, MediaHandler *outMediaHandler)
{
/*	Get the media identifier for the media that contains the first
	video track's sample data, and also get the media handler for
	this media.			*/
  *outMedia = NULL;
  *outMediaHandler = NULL;

  /* get first video track */
  Track videoTrack = GetMovieIndTrackType(inMovie, 1, kCharacteristicHasVideoFrameRate,
											movieTrackCharacteristic | movieTrackEnabledOnly);
  if (videoTrack != NULL)
  {
    /* get media ref. for track's sample data */
    *outMedia = GetTrackMedia(videoTrack);
    if (*outMedia)
    {
      /* get a reference to the media handler component */
      *outMediaHandler = GetMediaHandler(*outMedia);
    }
  }
}

bool IsMPEGMediaHandler(MediaHandler inMediaHandler)
{
	Boolean isMPEG;
	OSErr result = MediaHasCharacteristic(inMediaHandler,
											kCharacteristicIsAnMpegTrack,
											&isMPEG);
	ALWAYS_ASSERT(result == noErr);
	return (bool)isMPEG;
}

Fixed MPEGMediaGetStaticFrameRate(MediaHandler inMPEGMediaHandler)
{
	MHInfoEncodedFrameRateRecord encodedFrameRate;
	Size encodedFrameRateSize = sizeof(encodedFrameRate);

	ComponentResult err = MediaGetPublicInfo(inMPEGMediaHandler,
												kMHInfoEncodedFrameRate,
												&encodedFrameRate,
												&encodedFrameRateSize);
	ALWAYS_ASSERT(err == noErr);
	return encodedFrameRate.encodedFrameRate;
}

long JGetMovieNextInterestingTime(Movie inMovie, long currentTime)
{
	if (currentTime == -1)
		return -1;
		
	OSType	video = VideoMediaType;
	long	nextTime;
	Media	movieMedia;
	MediaHandler	movieMediaHandler;
	
	MovieGetVideoMediaAndMediaHandler(inMovie, &movieMedia, &movieMediaHandler);
	ALWAYS_ASSERT(movieMedia && movieMediaHandler);

	/* is this the MPEG-1/MPEG-2 media handler? */
	if (IsMPEGMediaHandler(movieMediaHandler))
	{
		Fixed rate = MPEGMediaGetStaticFrameRate(movieMediaHandler);
		double frameRate = Fix2X(rate);
		nextTime = currentTime + GetMovieTimeScale(inMovie) / frameRate;
		if (nextTime > GetMovieDuration(inMovie))
			return -1;
	}
	else
		GetMovieNextInterestingTime(inMovie, nextTimeStep, 1, &video, currentTime, fixed1, &nextTime, NULL);
		
	return nextTime;
}
