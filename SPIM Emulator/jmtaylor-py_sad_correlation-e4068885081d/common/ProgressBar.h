/*	ProgressBar.h
 
 Copyright 2011-2015 Jonathan Taylor. All rights reserved.
 
 Code to display a progress bar for time-consuming operations.
 TextualProgressBar writes a growing line to stdout
 CocoaProgressWindow (see separate file) displays a window showing a progress bar and
 indicating time taken, time remaining etc.
 */

#ifndef __PROGRESSBAR_H__
#define __PROGRESSBAR_H__

#include "jMutex.h"

class BaseProgressBar
{
  protected:
	double				currentProgress;
	double				length;
	static int			disabled;
	int				overrideEnabled;
	JMutex				progressMutex;

	double				startTime, startTimeForEstimate;
	double				startingProgressForEstimate;
	bool				reportElapsedTime;
	
	virtual void		InternalUpdateProgress(double newProgress) = 0;

  public:
						BaseProgressBar(double inLength);
	virtual				~BaseProgressBar();

	void				UpdateProgress(double newProgress);
	void				DeltaProgress(double delta);
	void				IncrementProgress(void) { DeltaProgress(1); }
	void				SetReportElapsedTime(bool rep) { reportElapsedTime = rep; }
	void				UpdateLength(double newLength);

	void				GetElapsedTime(int *hours, int *mins, double *secs);
	void				ResetTimeEstimate(void);
	void				EstimateTimeRemaining(int *hours, int *mins, double *secs);
	
	static void			Disable(void) { disabled++; }
	static void			Enable(void) { ALWAYS_ASSERT(disabled > 0); disabled--; }
	
	virtual void		OverrideEnable(void) { overrideEnabled = true; }
	bool				Enabled(void) const { return overrideEnabled || (disabled == 0); }
	double				Length(void) const { return length; }
};

class TextualProgressBar : public BaseProgressBar
{
  protected:
	static const char	*kTextProgressBarSpaces;
	static const int	kTextProgressBarWidth;
	int				numCharsDrawn;
	char				cachedTitle[256];
	bool				drawnTitle;

	virtual void		InternalUpdateProgress(double newProgress);
	void				SetTitle(const char *title, va_list argList);
	void				SetTitle(const char *title, ...) PRINTFLIKE(2, 3);

  public:
						TextualProgressBar(const char *title, double inLength, ...) PRINTFLIKE(2, 4);
	virtual				~TextualProgressBar() { if (Enabled()) printf("\n"); }
	virtual void		OverrideEnable(void);
};

typedef TextualProgressBar	ProgressBar;

#endif
