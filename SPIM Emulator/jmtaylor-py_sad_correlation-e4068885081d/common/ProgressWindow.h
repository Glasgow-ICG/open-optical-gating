/*	ProgressWindow.h
 
	Copyright 2010-2015 Jonathan Taylor. All rights reserved.

	OS X-only code which displays a window showing the progress of a time-consuming operation,
	printing the elapsed time and an estimate of the time remaining	until the operation is complete.
 */

#ifndef __PROGRESSWINDOW_H__
#define __PROGRESSWINDOW_H__

#include "ProgressBar.h"

#if 0//HAS_OS_X_GUI
// TODO: need to update this code to use CocoaProgressWindow - see also comments in .cpp file
class ProgressWindow : public BaseProgressBar
{
  protected:
	WindowRef		theWindow;
	int			windowWidth, windowHeight;
	
	virtual void	InternalUpdateProgress(double newProgress);

  public:
					ProgressWindow(int x, int y, const char *title, double inLength, ...) __attribute__ ((format (printf, 4, 6)));
	virtual			~ProgressWindow();
};
#else
class ProgressWindow : public TextualProgressBar
{
  public:
					ProgressWindow(int x, int y, const char *title, double inLength, ...) __attribute__ ((format (printf, 4, 6)));
	virtual			~ProgressWindow() { }
};
#endif

#endif
