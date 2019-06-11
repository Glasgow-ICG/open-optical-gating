//
//  jAlias.mm
//
//	Copyright 2010-2015 Jonathan Taylor. All rights reserved.
//
//	Cocoa class representing an alias for a filesystem object
//  (which should continue to resolve to the same object even if
//   it is moved around in the filesystem)
//

#import "jAlias.h"

@interface JAlias()
	@property (readwrite, copy) NSData *bookmark;
@end

@implementation JAlias

+(id)aliasForPath:(NSString *)path
{
	return [[[JAlias alloc] initForPath:path] autorelease];
}

-(id)initForPath:(NSString *)path
{
	return [self initForURL:[NSURL fileURLWithPath:path]];
}

+(id)aliasForURL:(NSURL *)url
{
	return [[[JAlias alloc] initForURL:url] autorelease];
}

-(id)initForURL:(NSURL *)url
{
	if (!(self = [super init]))
		return nil;
	NSError *err;
	self.bookmark = [url bookmarkDataWithOptions:0 includingResourceValuesForKeys:nil relativeToURL:nil error:&err];
	return self;
}

-(void)dealloc
{
	self.bookmark = nil;
	[super dealloc];
}

-(id)copyWithZone:(NSZone *)zone
{
	JAlias *copy = [[[self class] alloc] init];
	copy.bookmark = self.bookmark;
    return copy;
}

-(NSString *)path
{
	// Resolve the alias and return the current path to the object
	BOOL stale;
	NSError *error;
	NSURL *url = [NSURL URLByResolvingBookmarkData:self.bookmark options:NSURLBookmarkResolutionWithoutUI relativeToURL:nil bookmarkDataIsStale:&stale error:&error];
	return [url path];
}

-(NSURL *)url
{
	// Resolve the alias and return a current URL for the object
	BOOL stale;
	NSError *error;
	NSURL *url = [NSURL URLByResolvingBookmarkData:self.bookmark options:NSURLBookmarkResolutionWithoutUI relativeToURL:nil bookmarkDataIsStale:&stale error:&error];
	return url;
}

-(NSString *)filename
{
	// Return the filename for the object
	// Works even if bookmark is unresolvable
	NSDictionary *dict = [NSURL resourceValuesForKeys:[NSArray arrayWithObject:NSURLNameKey] fromBookmarkData:self.bookmark];
	return [dict objectForKey:NSURLNameKey];
}

-(BOOL)resolvesSameAs:(JAlias *)other
{
	// Check whether two alias objects refer to the same filesystem object
	return ([self.path compare:other.path] == NSOrderedSame);
}

@synthesize bookmark = _bookmark;

@end
