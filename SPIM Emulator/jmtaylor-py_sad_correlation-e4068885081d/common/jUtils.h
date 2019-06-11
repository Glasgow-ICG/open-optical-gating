//
//  jUtils.h
//
//	Copyright 2010-2015 Jonathan Taylor. All rights reserved.
//
//	A random assortment of utility functions!
//
#ifndef __JUTILS_H__
#define __JUTILS_H__

extern const double PI;
extern const double NaN;

namespace fundamental_constants
{
	extern const double c;
	extern const double e_0;
	extern const double mu_0;
	extern const double electronic_charge;
	extern const double eta_0;
	extern const double root4PiE0, root4PiMu0;
	extern const double k_b;
    extern const double g;
	extern const double L, lambda_b;
}

#if JREAL_DEFINED
extern const jreal PI_R;

namespace fundamental_constants_r
{
	extern const jreal c;
	extern const jreal e_0;
	extern const jreal mu_0;
	extern const jreal eta_0;
	extern const jreal root4PiE0, root4PiMu0;
}
#else
	extern const double PI_R;
	#define fundamental_constants_r fundamental_constants
#endif

#include "jCommon.h"
#include "jTimeUtils.h"
#include "jBigNum.h"
#include "jCoord.h"

void *void_aligned_malloc(size_t size, size_t align_size);
void aligned_free(volatile void *inPtr);
template<class C> C *aligned_malloc(size_t size, size_t align_size = 16)
{
	// A templated version of void_aligned_malloc that returns a pointer of a specified type
	return (C *)void_aligned_malloc(size * sizeof(C), align_size);
}

class LocalEnableDenormalFlushing
{
  protected:
	int	oldmxcsr;
	
  public:
	LocalEnableDenormalFlushing(void);
	~LocalEnableDenormalFlushing();
};

template<class C> void DeleteArrayIfNotNull(C *v)
{
	// Deletes the supplied memory (allocated using new[]) if the pointer is not NULL
	if (v != NULL)
		delete[] v;
}

template<class C> void DeleteIfNotNull(C *v)
{
	// Deletes the supplied memory (allocated using new) if the pointer is not NULL
	if (v != NULL)
		delete v;
}

const char *GetAddressString(int address, char addressString[128]);

inline double DegreesToRadians(double deg) { return deg / 180.0 * PI; }
inline double RadiansToDegrees(double rad) { return rad * 180.0 / PI; }

char *NewCopyOfString(const char *inString);

bool FileExists(const char *theFile);
FILE *fopenf(const char * RESTRICT format, const char * RESTRICT mode, ...) PRINTFLIKE(1, 3);

void LinearFit(std::vector<double> &x, std::vector<double> &y, double *alpha, double *beta);

#if __OBJC__
	#import <Cocoa/Cocoa.h>
    NSURL *PathToURL(NSString *path, NSURL *relativeTo);
    NSURL *PathToURL(NSString *path);
    bool IsDirectory(NSURL *fileURL);

	NSArray *ListImageFilesInDirectory(NSString *dir, bool sorted = true, bool useTimestamps = false/* Default to false just because this is slower and didn't used to be what I did*/, bool fullPath = false);
	void UpdateKeys(id owner, ...) NS_REQUIRES_NIL_TERMINATION;
	bool StringIsInList(NSString *s, ...) NS_REQUIRES_NIL_TERMINATION;

	typedef id (^BlockReturningObject)(void);
	@class MAZeroingWeakRef;
	id ResurrectWeakRef(MAZeroingWeakRef *&ref, BlockReturningObject resurrectionBlock);
	id ResurrectAndShowWeakWindowRef(MAZeroingWeakRef *&ref, BlockReturningObject resurrectionBlock);
  #ifdef __BLOCKS__
        void ForEveryImageFileInDirectory(NSString *dir, void (^callback)(NSString *));
        void ForEveryImageFileInDirectoryConcurrent(NSString *dir, void (^callback)(NSString *));
  #endif
    NSString *FirstImageFileNameInDirectory(NSString *dir);
	NSString *MetadataPathFromImagePath(NSString *fileName);
    id MetadataKeyValueForFramePath(NSString *path, NSString *key);
	void CopyMetadataForImageFile(NSString *sourceFilePath, NSString *destDirPath, NSString *destFileName = nil);
    void PrintCompleteFolderPath(NSString *basePath, int indentationLevel, int leadingCharsToSkip);
	NSInteger frameSortOrder(id string1, id string2, void *);
	NSInteger frameSortOrderForURLs(id url1, id url2, void *);
	NSInteger frameSortOrderUsingTimestamps(id string1, id string2, void *);
#endif

#endif
