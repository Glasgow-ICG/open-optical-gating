//
//  DictionaryReadingExtensions.mm
//
//	Copyright 2014-2015 Jonathan Taylor. All rights reserved.
//
//	Extensions to NSDictionary for easy and typesafe access to key values
//

#import "DictionaryReadingExtensions.h"

@implementation NSDictionary (KeyReading)

-(NSNumber *)getRequiredNumberForKey:(NSString *)key
{
	// Obtain the NSNumber associated with the specified key.
	// Will fail assertion if the key value is absent or is not an NSNumber.
	ALWAYS_ASSERT([self isKindOfClass:[NSDictionary class]]);
	NSNumber *num = [self objectForKey:key];
	ALWAYS_ASSERT([num isKindOfClass:[NSNumber class]]);
	return num;
}

-(NSNumber *)getOptionalNumberForKey:(NSString *)key defaultVal:(NSNumber *)def
{
	// Obtain the NSNumber associated with the specified key.
	// If the key is not present in the dictionary, return the supplied default value instead.
	// Will fail assertion if the key value is present but is not an NSNumber.
	ALWAYS_ASSERT([self isKindOfClass:[NSDictionary class]]);
	NSNumber *num = [self objectForKey:key];
    if (num == nil)
        return def;
	ALWAYS_ASSERT([num isKindOfClass:[NSNumber class]]);
	return num;
}

-(int)getRequiredIntForKey:(NSString *)key
{
    return [self getRequiredNumberForKey:key].intValue;
}

-(bool)getRequiredBoolForKey:(NSString *)key
{
    return [self getRequiredNumberForKey:key].boolValue;
}

-(double)getRequiredDoubleForKey:(NSString *)key
{
    return [self getRequiredNumberForKey:key].doubleValue;
}

-(NSString *)getRequiredStringForKey:(NSString *)key
{
	ALWAYS_ASSERT([self isKindOfClass:[NSDictionary class]]);
	NSString *str = [self objectForKey:key];
	ALWAYS_ASSERT([str isKindOfClass:[NSString class]]);
	return str;
}

-(NSArray *)getRequiredArrayForKey:(NSString *)key ofLength:(int)length
{
	ALWAYS_ASSERT([self isKindOfClass:[NSDictionary class]]);
	NSArray *arr = [self objectForKey:key];
	ALWAYS_ASSERT([arr isKindOfClass:[NSArray class]]);
	ALWAYS_ASSERT(arr.count == (NSUInteger)length);
	return arr;
}

-(NSArray *)getOptionalArrayForKey:(NSString *)key ofLength:(int)length defaultVal:(NSArray *)def
{
	ALWAYS_ASSERT([self isKindOfClass:[NSDictionary class]]);
	NSArray *arr = [self objectForKey:key];
	if (arr == nil)
		return def;
	ALWAYS_ASSERT([arr isKindOfClass:[NSArray class]]);
	if (length > 0)
		ALWAYS_ASSERT(arr.count == (NSUInteger)length);
	return arr;
}
-(NSDictionary *)getRequiredDictionaryForKey:(NSString *)key
{
	ALWAYS_ASSERT([self isKindOfClass:[NSDictionary class]]);
	NSDictionary *dict = [self objectForKey:key];
	ALWAYS_ASSERT([dict isKindOfClass:[NSDictionary class]]);
	return dict;
}

-(double)getOptionalDoubleForKey:(NSString *)key defaultVal:(double)def
{
    NSNumber *num = [self getOptionalNumberForKey:key defaultVal:[NSNumber numberWithDouble:def]];
	return num.doubleValue;
}

-(int)getOptionalIntForKey:(NSString *)key defaultVal:(int)def
{
    NSNumber *num = [self getOptionalNumberForKey:key defaultVal:[NSNumber numberWithInt:def]];
	return num.intValue;
}

-(bool)getOptionalBoolForKey:(NSString *)key defaultVal:(bool)def
{
    NSNumber *num = [self getOptionalNumberForKey:key defaultVal:[NSNumber numberWithBool:def]];
	return num.boolValue;
}

-(NSString *)getOptionalStringForKey:(NSString *)key defaultVal:(NSString *)def
{
	ALWAYS_ASSERT([self isKindOfClass:[NSDictionary class]]);
    NSString *str = [self objectForKey:key];
    if (str == nil)
        return def;
	ALWAYS_ASSERT([str isKindOfClass:[NSString class]]);
	return str;
}


@end
