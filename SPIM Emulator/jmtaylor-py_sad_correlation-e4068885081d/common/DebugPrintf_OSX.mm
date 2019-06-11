//
//  DebugPrintf_OSX.mm
//
//  Copyright 2015 Jonathan Taylor. All rights reserved.
//
//	Implementation of DebugPrintf suitable for running on OS X
//	Print to stderr but also call NSLog so that the message shows up even when running as a standalone app
//
//	Only one platform-specific implementation file like this one should be included in a project,
//	or else there will be linker errors due to multiple function definitions.
//

#include "DebugPrintf.h"
#include "jCommon.h"
#include <Cocoa/Cocoa.h>
#include <pthread.h>

void VDebugPrintf(const char *format, va_list args)
{
	va_list args2;
	__va_copy(args2, args);
	
	// Print to stderr
	vfprintf(stderr, format, args);
	
	// Also print to Apple's console log facility, to ensure it shows up when running standalone (not under debugger)
	NSLogv([SWF:@"%s", format], args2);
}

void DebugPrintf(const char *format, ...)
{
	va_list args;

	// Print to stderr
	va_start(args, format);
	VDebugPrintf(format, args);
	va_end(args);
}

void DebugPrintfFatal(const char *errorIntro, const char *format, ...)
{
	va_list args;
	
	va_start(args, format);
	VDebugPrintf(format, args);
	va_end(args);

	va_start(args, format);
    NSAutoreleasePool *pool = [NSAutoreleasePool new];
    NSString *errorDetails = [[NSString alloc] initWithFormat:[SWF:@"%s", format] arguments:args];
	dispatch_block_t theBlock = ^{
		NSAlert *alert = [NSAlert alertWithMessageText:[SWF:@"%s", errorIntro]
										 defaultButton:@"OK"
									   alternateButton:nil
										   otherButton:nil
							 informativeTextWithFormat:@"%@", errorDetails];
		[alert runModal];
	};
	if (pthread_main_np())
		theBlock();
	else
		dispatch_sync(dispatch_get_main_queue(), theBlock);
    [pool drain];
	va_end(args);
}
