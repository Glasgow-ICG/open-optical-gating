//
//  jApplication.h
//  Simple Preview
//
//  Created by Jonathan Taylor on 10/9/15.
//  Copyright 2015 Jonathan Taylor. All rights reserved.
//

#import <Cocoa/Cocoa.h>

@interface JApplication : NSApplication
{
	bool		_terminating;
	NSString	*_buildVersionString;
	NSString            *_configFilename;
}

-(void)applicationDidFinishLaunching:(NSNotification *)note;
-(void)alertWithText:(NSString *)mainText andExplanation:(NSString *)text2;
-(void)alertWithText:(NSString *)mainText andExplanation:(NSString *)text2 iconName:(NSString*)iconName;

-(void)addGlobalMetadataToDictionary:(NSMutableDictionary *)dict;

-(void)setObject:(id)val forDefault:(NSString *)key;
-(id)getObjectForDefault:(NSString *)key requiringClass:(Class)c mayBeAbsent:(bool)mayBeAbsent;
-(NSString *)stringForDefault:(NSString *)key;
-(NSString *)stringForDefault:(NSString *)key mayBeAbsent:(bool)mayBeAbsent;
-(NSString *)stringForDefault:(NSString *)key usingIfAbsent:(NSString *)def;
-(NSDictionary *)dictionaryForDefault:(NSString *)key mayBeAbsent:(bool)mayBeAbsent;
-(int)intForDefault:(NSString *)key;
-(int)intForDefault:(NSString *)key usingIfAbsent:(int)def;
-(double)doubleForDefault:(NSString *)key;

@property (readonly, retain) NSString *configFilename;
@property (readonly, retain) NSString *buildVersionString;
@property (readonly) bool debugBuild;
@property (readonly) bool terminating;

@end

extern JApplication *baseApp;
