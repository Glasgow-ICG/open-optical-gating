//
//  JWindowController.h
//
//	Copyright 2011-2015 Jonathan Taylor. All rights reserved.
//
//  Subclass of NSWindowController that helps with notification whenever windows are opened or closed
//

#import <Cocoa/Cocoa.h>

extern NSMutableSet *activeWindowControllers;
extern NSString *WindowControllerListChanged;

@interface JWindowController : NSWindowController <NSWindowDelegate>
{
    int _windowControllerListChanged_dummyProperty;
}
@property (readonly) int windowControllerListChanged_dummyProperty;
@end
