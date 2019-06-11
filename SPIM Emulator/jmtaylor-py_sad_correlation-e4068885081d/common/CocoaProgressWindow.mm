//
//  CocoaProgressWindow.mm
//
//  Copyright 2010-2015 Jonathan Taylor. All rights reserved.
//
//	A cocoa wrapper for my BaseProgressBar class, to display a progress window as a proper OS X window
//

#import "CocoaProgressWindow.h"
#import "ProgressBar.h"
#import "jNotifications.h"

#ifndef SWF
#define SWF NSString stringWithFormat
#endif

class CocoaProgressWindowHelper : public BaseProgressBar
{
	// Helper class that allows the InternalUpdateProgress function in BaseProgressBar
	// to call through to the actual display code for the Cocoa window
  protected:
	CocoaProgressWindow *windowClass;
	virtual void		InternalUpdateProgress(double newProgress);
	
  public:
	CocoaProgressWindowHelper(double inLength, CocoaProgressWindow *win)
			: BaseProgressBar(inLength), windowClass(win) { }
};

void CocoaProgressWindowHelper::InternalUpdateProgress(double newProgress)
{
	currentProgress = MIN(newProgress, length);	
	[windowClass internalUpdateProgress:currentProgress];
}

@implementation CocoaProgressWindow

-(id)initForItems:(double)inLength withTitle:(NSString *)title
{
	return [self initForItems:inLength withTitle:title sheetOnWindow:nil];
}

-(id)initForItems:(double)inLength withTitle:(NSString *)title sheetOnWindow:(NSWindow *)win
{
	if (inLength == 0)
		return [self initIndeterminateWithTitle:title sheetOnWindow:win];
	
	if (!(self = [self initWithWindowNibName:@"ProgressPanel"]))
		return nil;
	self.window.title = title;
	_base = new CocoaProgressWindowHelper(inLength, self);
	self.progressCaption = title;
	sheetBegun = false;
	[indicator setMaxValue:inLength];

	if (win != nil)
		[self setUpSheetOnWindow:win];
	
	return self;
}

-(id)initInitiallyIndeterminateWithTitle:(NSString *)title sheetOnWindow:(NSWindow *)win
{
	if (!(self = [self initWithWindowNibName:(win ? @"ProgressPanel" : @"ProgressWindow")]))
		return nil;
	self.window.title = title;
	_base = new CocoaProgressWindowHelper(0, self);
	self.progressCaption = title;

	sheetBegun = false;
	[indicator setIndeterminate:YES];
	[indicator startAnimation:nil];
	
	if (win != nil)
		[self setUpSheetOnWindow:win];
	
	return self;
}

-(id)initIndeterminateWithTitle:(NSString *)title sheetOnWindow:(NSWindow *)win
{
	if (!(self = [self initWithWindowNibName:@"ProgressPanelIndeterminate"]))
		return nil;
	self.window.title = title;
	_base = new CocoaProgressWindowHelper(0, self);
	self.progressCaption = title;
	sheetBegun = false;
	[indicator startAnimation:nil];

	if (win != nil)
		[self setUpSheetOnWindow:win];
	return self;
}

-(id)initIndeterminateOverlayWithTitle:(NSString *)title withControl:(NSProgressIndicator *)inProgressIndicator
{
	if (!(self = [self initWithWindow:nil]))
		return nil;
	_base = new CocoaProgressWindowHelper(0, self);
	indicator = inProgressIndicator;
	self.progressCaption = title;
	sheetBegun = false;
	[indicator startAnimation:nil];
	return self;
}

-(void)dealloc
{
	printf("Dealloc progress window %p\n", self);
	CHECK(!sheetBegun);		// Owner should call closeSheetAndRelease
	delete _base;
	[super dealloc];
}

-(void)upgradeToDeterminateLength:(double)inLength
{
	_base->UpdateLength(inLength);
	[indicator setIndeterminate:(inLength == 0)];
	[indicator setMaxValue:inLength];
	[indicator startAnimation:nil];
	_base->ResetTimeEstimate();
}

-(void)closeWindowAndRelease
{
	[[self window] orderOut:nil];
	[[self window] close];
	[self release];
}

-(void)closeSheetAndRelease
{
	if (sheetBegun)
	{
		[NSApp endSheet:[self window] returnCode:NSOKButton];
		[[self window] orderOut:nil];
		[[self window] close];
		sheetBegun = false;
	}
	[self release];
}

-(void)setUpSheetOnWindow:(NSWindow *)win
{
	[NSApp beginSheet:[self window] 
				modalForWindow:win
				modalDelegate:win 
				didEndSelector:nil
				contextInfo:nil];
	sheetBegun = true;

	[[NSNotificationCenter defaultCenter] addObserver:self selector:@selector(closeSheetAndRelease) name:CloseSheetsForTermination object:nil];
}

-(void)internalUpdateProgress:(double)newProgress
{
	// This currently updates for every single delta progress.
	// TODO: I should instigate some sort of flow control in here
	// Possibly just do the updating based on a timer.
	[indicator setDoubleValue:newProgress];
	[fraction setStringValue:[SWF:@"%d/%d", (int)newProgress, (int)_base->Length()]];

	int hours, mins;
	double secs;
	_base->GetElapsedTime(&hours, &mins, &secs);
	[elapsed setStringValue:[SWF:@"%dh%dm%.02lfs",hours, mins, secs]];
	_base->EstimateTimeRemaining(&hours, &mins, &secs);
	[remaining setStringValue:[SWF:@"%dh%dm%.02lfs",hours, mins, secs]];
	
	[fraction setNeedsDisplay];
	[elapsed setNeedsDisplay];
	[remaining setNeedsDisplay];
}

-(void)resetTimeEstimate
{
	_base->ResetTimeEstimate();
}

-(double)progressValue
{
	return indicator.doubleValue;
}

-(void)setProgressValue:(double)val
{
	_base->UpdateProgress(val);
}

-(void)deltaProgress:(double)delta
{
	_base->DeltaProgress(delta);
}

-(IBAction)cancel:(id)sender
{
	// The code that is managing this progress bar must poll for cancellation
	self.userCancelled = true;
}

@synthesize progressCaption = _progressCaption;
@synthesize userCancelled = _userCancelled;

@end
