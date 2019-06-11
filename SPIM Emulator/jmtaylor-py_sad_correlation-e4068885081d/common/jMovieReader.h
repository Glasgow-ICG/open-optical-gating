/*
 *  JMovieReader.h
 *
 *  Created by Jonathan Taylor on 13/06/2009.
 *  Copyright 2009 Durham University. All rights reserved.
 *
 */

#ifndef __J_MOVIE_READER_H__
#define __J_MOVIE_READER_H__ 1

class JMovieReader
{
  protected:
	WindowRef	theWindow;
	Movie		theMovie;
	Rect		movieRect;
	char		movieFilename[256];
	TimeValue	currentTime;
  public:
				JMovieReader(Point windowOrigin = (Point){-1, -1}, const char *path = NULL);
				JMovieReader(Point windowOrigin, const FSSpec *spec);
	bool		MaybeCreateWindow(Point &windowOrigin);
	GWorldPtr	NewDestGWorld(void);
	const Rect	*MovieRect(void) const { return &movieRect; }
	WindowRef	MovieWindow(void) { return theWindow; }
	long		EstimatedFrameCount(void);
	TimeValue	SkipFrame(void);
	TimeValue	DrawNextFrameIntoGWorld(GWorldPtr frameGWorld);
};

#endif
