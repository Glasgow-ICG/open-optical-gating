//
//  jAlias.h
//
//	Copyright 2010-2015 Jonathan Taylor. All rights reserved.
//
//	Cocoa class representing an alias for a filesystem object
//  (which should continue to resolve to the same object even if
//   it is moved around in the filesystem)
//

#import <Cocoa/Cocoa.h>


@interface JAlias : NSObject {
	NSData *_bookmark;
}
@property (readonly) NSString *path;
@property (readonly) NSURL *url;
@property (readonly) NSString *filename; // Works even if bookmark is unresolvable

+(id)aliasForPath:(NSString *)path;
+(id)aliasForURL:(NSURL *)url;
-(id)initForPath:(NSString *)path;
-(id)initForURL:(NSURL *)url;
-(BOOL)resolvesSameAs:(JAlias *)other;

@end
