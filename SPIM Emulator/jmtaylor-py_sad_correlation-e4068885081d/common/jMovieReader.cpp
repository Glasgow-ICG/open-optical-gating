/*
 *  JMovieReader.cpp
 *
 *  Created by Jonathan Taylor on 13/06/2009.
 *  Copyright 2009 Durham University. All rights reserved.
 *
 */

#include "JMovieReader.h"
#include "DTSQTUtilities.h"
#include "jAssert.h"

const long kMaxMillisecs = 0;	// Timeout for MoviesTask

JMovieReader::JMovieReader(Point windowOrigin, const FSSpec *spec)
{
	EnterMovies();
	
	OSErr anErr = QTUSimpleGetMovieFromFSSpec(&theMovie, movieFilename, spec);
	ALWAYS_ASSERT_NOERR(anErr);
	currentTime = 0;

	// Create a window if required
	MaybeCreateWindow(windowOrigin);
}

JMovieReader::JMovieReader(Point windowOrigin, const char *path)
{
	EnterMovies();
	
	if (path != NULL)
	{
		// Mac-style path can be provided for GUI-less operation
		// e.g. "Development:experiment_videos:Disk 4:MediaBrowserFile_005A89AECF003000.mpg"
		// e.g. "Macintosh HD:Users:jonny:Documents:Disk 5:MediaBrowserFile_00192797A3003000.mpg"
		OSErr anErr = QTUSimpleGetMovieFromPath(&theMovie, movieFilename, path);
		ALWAYS_ASSERT_NOERR(anErr);
	}
	else
	{
		// Prompt the user to select a movie file
		OSErr anErr = QTUSimpleGetMovie(&theMovie, movieFilename);
		ALWAYS_ASSERT_NOERR(anErr);
	}
	currentTime = 0;

	// Create a window if required
	MaybeCreateWindow(windowOrigin);
}

bool JMovieReader::MaybeCreateWindow(Point &windowOrigin)
{
	if (windowOrigin.h != -1)
	{
		// Size the window to fit the movie
		Rect 		windowBounds;
		GetMovieBox(theMovie, &movieRect);
		GetMovieBox(theMovie, &windowBounds);
		OffsetRect(&windowBounds, windowOrigin.h - windowBounds.left, windowOrigin.v - windowBounds.top);

		// Create the window we will use.
		OSErr anErr = CreateNewWindow( kDocumentWindowClass, 
									kWindowCloseBoxAttribute, 
									&windowBounds, 
									&theWindow );
		ALWAYS_ASSERT_NOERR(anErr);
		ShowWindow(theWindow);
		SetPortWindowPort(theWindow);
		return true;
	}
	else
	{
		theWindow = NULL;
		return false;
	}
}

GWorldPtr JMovieReader::NewDestGWorld(void)
{
	// Create a gWorld matching the movie dimensions
	GWorldPtr		theGWorld;
	CGrafPtr		aSavedPort;
	GDHandle		aSavedGDevice;
	GetGWorld(&aSavedPort, &aSavedGDevice);
	CTabHandle		colorTable = (**(**aSavedGDevice).gdPMap).pmTable;
	OSErr anErr = NewGWorld(&theGWorld, 32, &movieRect, colorTable, NULL, 0);
	ALWAYS_ASSERT_NOERR(anErr);
	return theGWorld;
}

long JMovieReader::EstimatedFrameCount(void)
{
	// We assume constant frame durations. Divide movie duration by duration of first frame
	long firstFrameTime = JGetMovieNextInterestingTime(theMovie, 0);
	return GetMovieDuration(theMovie) / firstFrameTime;
}

TimeValue JMovieReader::SkipFrame(void)
{
	TimeValue nextTime = JGetMovieNextInterestingTime(theMovie, currentTime);
	currentTime = nextTime;
	return JGetMovieNextInterestingTime(theMovie, currentTime);
}

TimeValue JMovieReader::DrawNextFrameIntoGWorld(GWorldPtr frameGWorld)
{
	TimeValue nextTime = JGetMovieNextInterestingTime(theMovie, currentTime);

	SetMovieGWorld(theMovie, frameGWorld, GetGWorldDevice(frameGWorld));
	SetMovieTimeValue(theMovie, currentTime); 
	UpdateMovie(theMovie);
	MoviesTask(theMovie, kMaxMillisecs);
	currentTime = nextTime;
	return nextTime;
}
