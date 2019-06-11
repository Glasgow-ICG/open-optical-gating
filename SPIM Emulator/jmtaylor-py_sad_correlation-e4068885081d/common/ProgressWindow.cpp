/*	ProgressWindow.cpp

	Copyright 2010-2015 Jonathan Taylor. All rights reserved.

	OS X-only code which displays a window showing the progress of a time-consuming operation,
	printing the elapsed time and an estimate of the time remaining	until the operation is complete.	*/

#include "ProgressWindow.h"
#include "jUtils.h"

#if 0//HAS_OS_X_GUI
/* As currently written, this code doesn't work on latest OS versions where Carbon is unavailable.
    Need to update this code to CocoaProgressWindow 
    There may also be issues around (maybe) not having a proper run loop, and yet wanting the window
    to appear, animate etc. That may need a bit of looking into
 */

ProgressWindow::ProgressWindow(int x, int y, const char *title, double inLength, ...) : BaseProgressBar(inLength)
{
	ControlRef	rootControl;
	Str255 buffer;
	
	va_list		argList;
	va_start(argList, inLength);
	
	windowWidth = 200;
	windowHeight = 100;
	
	Rect		windowRect;
	OSStatus	result;
	SetRect(&windowRect, x, y, x + windowWidth, y + windowHeight);
	result = CreateNewWindow(kDocumentWindowClass,
							 (kWindowStandardDocumentAttributes | kWindowStandardHandlerAttribute) & (~kWindowResizableAttribute),
							 &windowRect,
							 &theWindow);
	ASSERT_NOERR(result);
	CreateRootControl(theWindow, &rootControl);	

	SetPort(GetWindowPort(theWindow));
	TextSize(9);
	SetWTitle(theWindow, ConvertCToPascalString(title, buffer));
	ShowWindow(theWindow);
}

ProgressWindow::~ProgressWindow()
{
	DisposeWindow(theWindow);
}

void ProgressWindow::InternalUpdateProgress(double newProgress)
{
	currentProgress = MIN(newProgress, length);
	
	double	fraction = currentProgress / length;
	char	buffer[1024];
	double curTime;
	
	SetPort(GetWindowPort(theWindow));
	ForeColor(33);
	
	Rect	theRect;
	SetRect(&theRect, 0, 0, (short)(fraction * windowWidth), 30);
	PaintRect(&theRect);
	
	SetRect(&theRect, 0, 30, windowWidth, windowHeight);
	EraseRect(&theRect);
	
	// Print progress
	MoveTo(10, 50);
	snprintf(buffer, sizeof(buffer), "%.0lf/%.0lf", currentProgress, length);
	DrawText(buffer, 0, strlen(buffer));
	
	int hours, mins;
	double secs;
	
	// Print time elapsed
	GetElapsedTime(&hours, &mins, &secs);
	snprintf(buffer, sizeof(buffer), "%dh%dm%.02lfs", hours, mins, secs);
	MoveTo(120, 50);
	DrawText(buffer, 0, strlen(buffer));
	
	// Estimate time remaining
	EstimateTimeRemaining(&hours, &mins, &secs);
	snprintf(buffer, sizeof(buffer), "%dh%dm%.02lfs", hours, mins, secs);
	MoveTo(120, 80);
	DrawText(buffer, 0, strlen(buffer));

	QDFlushPortBuffer(GetWindowPort(theWindow), NULL);
}

#else

// Fall back to TextualProgressBar!

ProgressWindow::ProgressWindow(int x, int y, const char *title, double inLength, ...) : TextualProgressBar(NULL, inLength)
{
	va_list		argList;
	va_start(argList, inLength);
	SetTitle(title, argList);
}

#endif
