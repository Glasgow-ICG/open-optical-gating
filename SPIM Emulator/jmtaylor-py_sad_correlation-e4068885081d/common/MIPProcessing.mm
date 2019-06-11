//
//  MIPProcessing.mm
//  Spim Interface
//
//  Created by Jonny Taylor on 10/09/2015.
//
//

#import "MIPProcessing.h"
#import "CocoaProgressWindow.h"
#import <dispatch/dispatch.h>
#import "jCocoaImageUtils.h"
#ifdef __SSE4_1__
    #import <smmintrin.h>		// SSE4.1
#endif

dispatch_queue_t processingQueue = dispatch_queue_create("mip processing queue", NULL);

#ifndef __SSE4_1__
static inline __m128i _mm_blendv_si128 (__m128i x, __m128i y, __m128i mask)
{
	// Replace bit in x with bit in y when matching bit in mask is set:
	return _mm_or_si128(_mm_andnot_si128(mask, x), _mm_and_si128(mask, y));
}

static inline __m128i _mm_cmple_epu16 (__m128i x, __m128i y)
{
	// Returns 0xFFFF where x <= y:
	return _mm_cmpeq_epi16(_mm_subs_epu16(x, y), _mm_setzero_si128());
}
#endif

#define XY_MIP 1
#define XZ_MIP 2
#define MIP_TYPE XY_MIP

template<class PIX_TYPE> void CalcMipScalar(PIX_TYPE *mipPixels, const PIX_TYPE *otherPixels, size_t numPixels)
{
	// Old scalar code for reference
	for (size_t x = 0; x < numPixels; x++)
		mipPixels[x] = MAX(mipPixels[x], otherPixels[x]);
}

void CalcMip(unsigned char *mipPixels, const unsigned char *otherPixels, size_t numPixels)
{
#if 0
	CalcMipScalar(mipPixels, otherPixels, numPixels);
#else
	/* Vectorized MIP code.
	 
	 Basically this is not normally the bottleneck when image files need to be loaded from disk.
	 I'm not sure what scenario I had previously looked at, where this was a performance bottleneck!
	 */
	size_t i;
	for (i = 0; i < numPixels; i += 16)
	{
		__m128i x = *(__m128i*)&mipPixels[i];
		__m128i y = *(__m128i*)&otherPixels[i];
		*((__m128i*)&mipPixels[i]) = _mm_max_epu8(x, y);
	}
	for (; i < numPixels; i++)
		mipPixels[i] = MAX(mipPixels[i], otherPixels[i]);
#endif
}

void CalcMip(unsigned short *mipPixels, const unsigned short *otherPixels, size_t numPixels)
{
#if 0
	CalcMipScalar(mipPixels, otherPixels, numPixels);
#elif __SSE4_1__
	// Vectorized MIP code using SSE4.1 instruction set
	// On the macbook air (when manually enabled) this is only perhaps 5-10% faster than the code below.
	size_t i;
	for (i = 0; i < numPixels; i += 8)
		*((__m128i*)&mipPixels[i]) = _mm_max_epu16(*(__m128i*)&mipPixels[i], *(__m128i*)&otherPixels[i]);
	for (; i < numPixels; i ++)
		mipPixels[i] = MAX(mipPixels[i], otherPixels[i]);
#else
	/* Vectorized MIP code. This uses substitute code emulating _mm_max_epu16,
	 cribbed from http://www.alfredklomp.com/programming/sse-intrinsics/
	 for systems that do not support SSE4.1. At the time of writing, this
	 is true of most of the systems I am running the code on,
	 and it seems that even on the macbook air these are not enabled by default,
	 even though they are available.
	 
	 This version here runs at least 3x as fast as the scalar code on the macbook air,
	 and 4x as fast on the mac pro. These results are true when the source images are cached from disk.
	 When reading from disk, that is unsurprisingly the bottleneck and the net gains are less.
	 */
	size_t i;
	for (i = 0; i < numPixels; i += 8)
	{
		__m128i x = *(__m128i*)&mipPixels[i];
		__m128i y = *(__m128i*)&otherPixels[i];
		*((__m128i*)&mipPixels[i]) = _mm_blendv_si128(x, y, _mm_cmple_epu16(x, y));
	}
	for (; i < numPixels; i++)
		mipPixels[i] = MAX(mipPixels[i], otherPixels[i]);
#endif
}

void CalcMipForBPP(unsigned char *mipData, const unsigned char *otherData, size_t bytes, int bitsPerPixel)
{
	switch (bitsPerPixel)
	{
		case 8:
		case 32:
			// In the case of 32-bit data, we can treat it just as if it's 8-bit greyscale, but with more pixels
			CalcMip(mipData, otherData, bytes);
			break;
		case 16:
			CalcMip((unsigned short *)mipData, (const unsigned short*)otherData, bytes/2);
			break;
		default:
			ALWAYS_ASSERT(0);
	};
}

void MakeMipFromImagesInFolder(NSString *sourceFolderPath, NSString *destFilename, CocoaProgressWindow *progress, double totalWork)
{
	// TODO: this could be improved on - currently it just bails out if the files aren't in the format it expects
	NSAutoreleasePool *pool = [NSAutoreleasePool new];

	size_t numImages = ListImageFilesInDirectory(sourceFolderPath).count;
	NSBitmapImageRep *firstBitmap = RawBitmapFromImagePath(FirstImageFileNameInDirectory(sourceFolderPath));
	printf(" First file: %s %p\n", FirstImageFileNameInDirectory(sourceFolderPath).UTF8String, firstBitmap);

	__block int counter = 0;
	double shear = 0.0;//2.0;
	int shearStart = 0;//-100 * shear;
#if MIP_TYPE == XY_MIP
	__block NSBitmapImageRep *mipBitmap = [firstBitmap retain];
	unsigned char *mipData = mipBitmap.bitmapData;
	memset(mipData, 0, mipBitmap.bytesPerRow * mipBitmap.pixelsHigh);
	ForEveryImageFileInDirectory(sourceFolderPath,
								 ^(NSString *filename){
									 if (!progress.userCancelled)		// Not sure we can break out of a block-based loop, but we can bail fairly quickly like this
									 {
										 NSBitmapImageRep *otherBitmap = RawBitmapFromImagePath(filename);
										 if (!(CHECK(otherBitmap != nil)))
											 return;
                                         ALWAYS_ASSERT(otherBitmap != nil);     // Redundant, but useful to silence spurious static analysis warning
										 if (!(CHECK(otherBitmap.bytesPerRow == mipBitmap.bytesPerRow)))
											 return;
										 if (!(CHECK(otherBitmap.pixelsHigh == mipBitmap.pixelsHigh)))
											 return;
#if 0
										 // Optional compile-time feature:
										 // check plist to see if a triggered frame is reasonably close to the target phase
										 NSMutableDictionary *metadata = [NSMutableDictionary dictionaryWithContentsOfFile:MetadataPathFromImagePath(filename)];
										 NSNumber *err = [metadata objectForKey:@"estimated_ref_frame_error"];
										 if ((err != nil) &&
											 (fabs(err.doubleValue) > 5.0))
										 {
											 return;
										 }
#endif
										 if (shear == 0.0)
											 CalcMipForBPP(mipData, otherBitmap.bitmapData, size_t(mipBitmap.pixelsHigh * mipBitmap.bytesPerRow), (int)mipBitmap.bitsPerPixel);
										 else
										 {
											 if (!(CHECK(otherBitmap.bitsPerPixel == 16)))
											 {
												 // Not supported here yet.
												 // To be honest, this shear code is probably obsolete now I am supporting rotation in StackViewer
												 return;
											 }
											 for (int y = 0; y < otherBitmap.pixelsHigh; y++)
											 {
												 unsigned short *mipRow = (unsigned short *)(mipBitmap.bitmapData + mipBitmap.bytesPerRow * y);
												 int yToUse = y + shearStart + int(shear * counter);
												 if ((yToUse >= 0) && (yToUse < otherBitmap.pixelsHigh))
												 {
													 const unsigned short *otherRow = (const unsigned short *)(otherBitmap.bitmapData + otherBitmap.bytesPerRow * yToUse);
													 // No obvious way to do non-scalar, due to the fact that bytesPerRow may not be sufficiently aligned
													 // for the SSE instructions we would use for a vectorized implementation.
													 CalcMipScalar(mipRow, otherRow, otherBitmap.pixelsWide);
												 }
											 }
										 }
										 counter++;
										 dispatch_async(dispatch_get_main_queue(), ^{ [progress deltaProgress:(totalWork / double(numImages))]; });
									 }
								 });
#elif MIP_TYPE == XZ_MIP
	__block NSBitmapImageRep *mipBitmap = [[NSBitmapImageRep alloc]
											initWithBitmapDataPlanes:NULL		// Bitmap allocates and releases the necessary memory for us
											pixelsWide:firstBitmap.pixelsWide
											pixelsHigh:numImages
											bitsPerSample:16
											samplesPerPixel:1
											hasAlpha:NO
											isPlanar:NO
											colorSpaceName:NSCalibratedWhiteColorSpace
											bytesPerRow:firstBitmap.bytesPerRow
											bitsPerPixel:0];
	ForEveryImageFileInDirectory(sourceFolderPath,
								 ^(NSString *filename){
									 if (!progress.userCancelled)		// Not sure we can break out of a block-based loop, but we can bail fairly quickly like this
									 {
										 NSBitmapImageRep *otherBitmap = RawBitmapFromImagePath(filename);
										 if (!(CHECK(otherBitmap != nil)))
											 return;
										 if (!(CHECK(otherBitmap.bitsPerPixel == 16)))
											 return;
										 if (!(CHECK(otherBitmap.bytesPerRow == mipBitmap.bytesPerRow)))
											 return;
										 ALWAYS_ASSERT(counter < mipBitmap.pixelsHigh);
										 unsigned short *mipRow = (unsigned short *)(mipBitmap.bitmapData + mipBitmap.bytesPerRow * counter);
										 for (int y = 0; y < otherBitmap.pixelsHigh; y++)
										 {
											 int yToUse = y + shearStart + int(shear * counter);
											 if ((yToUse >= 0) && (yToUse < otherBitmap.pixelsHigh))
											 {
												 const unsigned short *otherRow = (const unsigned short *)(otherBitmap.bitmapData + otherBitmap.bytesPerRow * yToUse);
												 // No obvious way to do non-scalar, due to the fact that bytesPerRow may not be sufficiently aligned
												 // for the SSE instructions we would use for a vectorized implementation.
												 CalcMipScalar(mipRow, otherRow, otherBitmap.pixelsWide);
											 }
										 }
										 counter++;
										 dispatch_async(dispatch_get_main_queue(), ^{ [progress deltaProgress:(totalWork / double(numImages))]; });
									 }
								 });
#endif
	
	[[mipBitmap TIFFRepresentation] writeToFile:destFilename atomically:NO];
	printf(" Saving as %s\n", destFilename.UTF8String);
	[mipBitmap release];
	[pool drain];
}

void ProcessStacksIntoMIPsSavingAt(NSArray *inURLs, NSURL *destinationURL, void (^completionBlock)(int mipCounter, NSURL *destinationURL))
{
	// Sort filenames in chronological order
	NSArray *urls = [inURLs sortedArrayUsingFunction:frameSortOrderForURLs context:nil];
	CocoaProgressWindow *progress = [[CocoaProgressWindow alloc] initForItems:urls.count
																	withTitle:@"Generating MIPs..."
																sheetOnWindow:nil];
	[progress.window makeKeyAndOrderFront:nil];
	// We need to run this on a separate queue or we will encounter deadlocks
	// MovieBuilder expects to be able to dispatch_sync to the main queue
	dispatch_async(processingQueue,
				   ^{
					   int stackCounter = 0;
					   int mipCounter = 0;
					   for (NSURL *url in urls)
					   {
						   printf("File %s\n", url.path.UTF8String);
						   NSAutoreleasePool *pool = [NSAutoreleasePool new];
						   
						   // Process each of the image folders contained within the stack folder
						   NSString *mipFilename = [SWF:@"mip_%04d.tif", stackCounter++];
						   NSArray *dirContents = [[NSFileManager defaultManager] contentsOfDirectoryAtPath:url.path error:nil];
						   
						   // First do a rough estimate of number of camera folders present (for the progress bar)
						   int numCamFolders = 0;
						   for (NSString *cameraFolderPath in dirContents)
						   {
							   NSURL *contentsURL = PathToURL(cameraFolderPath, url);
							   if ((IsDirectory(contentsURL)) &&
								   (FirstImageFileNameInDirectory(contentsURL.path) != nil))
							   {
								   numCamFolders++;
							   }
						   }
						   if (FirstImageFileNameInDirectory(url.path) != nil)
						   {
							   // We also (or, more likely, instead...) have image files in the base directory itself
							   numCamFolders++;
							   // Add the base directory in, in order to pick up those image files too
							   dirContents = [dirContents arrayByAddingObject:@"."];
						   }
						   
						   // Now iterate properly, processing the files
						   for (NSString *cameraFolderPath in dirContents)
						   {
							   NSString *cameraFolderName = cameraFolderPath.lastPathComponent;
							   NSURL *contentsURL = PathToURL(cameraFolderPath, url);
							   if ((IsDirectory(contentsURL)) &&
								   (FirstImageFileNameInDirectory(contentsURL.path) != nil))
							   {
								   // This is a genuine camera folder containing images.
								   // Process a MIP for them, saving it into the appropriate folder
								   NSError *err;
								   NSString *destDirPath = [SWF:@"%@/%@", destinationURL.path, cameraFolderName];
								   BOOL ok = [[NSFileManager defaultManager] createDirectoryAtPath:destDirPath
																	   withIntermediateDirectories:YES
																						attributes:nil
																							 error:&err];
								   CHECK(ok);
								   
								   NSString *destFilename = [SWF:@"%@/%@", destDirPath, mipFilename];
								   NSString *destMetadataName = [SWF:@"%@.plist", destFilename.lastPathComponent.stringByDeletingPathExtension];
								   MakeMipFromImagesInFolder(contentsURL.path, destFilename, progress, 1.0 / numCamFolders);
								   CopyMetadataForImageFile(FirstImageFileNameInDirectory(contentsURL.path), destDirPath, destMetadataName);
								   mipCounter++;
								   if (progress.userCancelled)
									   break;
							   }
						   }
						   
#if 0
						   // Alternative code to generate a single sequence of frames that can be imported into ImageJ and turned into a hyperstack
						   // So far there isn't a GUI for this (need to make sure there is a fixed number of frames in each stack, discarding excess
						   // if necessary, offer a means of tweaking gain/offsets etc that is applied to all stacks that are processed, etc)
						   GUIMovieBuilder *builder = [[GUIMovieBuilder alloc] initAndRunBackgroundSession:@"Movie Builder"];
						   [builder addSequenceUsingDirectoryURL:url];
						   builder.tiffStem = [SWF:@"stack_%04d_", counter++];
						   
						   builder.maskEnabled = true;
						   builder.mask = [JRect rectWithNSRect:NSMakeRect(0, 0, 1000, 1000)];
						   builder.endFrame = 50;		// TODO: temp hack to deal with inconsistent frame counts in each stack
						   if (builder.sequences.count == 2)
							   [[builder.sequences objectAtIndex:1] setExposure:60];
						   
						   [builder saveMovie:nil andTiffs:destURL andMIP:nil];
						   [builder blockWhileBusy];
						   [builder release];
#endif
						   // We updateProgress rather than deltaProgress because we don't know
						   // what the individual stack-processing code may do in terms of delta progress
						   dispatch_async(dispatch_get_main_queue(), ^{ progress.progressValue = stackCounter; });
						   [pool drain];
						   if (progress.userCancelled)
							   break;
					   }
					   
					   
					   dispatch_async(dispatch_get_main_queue(), ^{
							   if (!progress.userCancelled)
								   completionBlock(mipCounter, destinationURL);
							   [progress closeWindowAndRelease];
					   });
				   });
}

void ProcessTheseStacksIntoMIPs(NSArray *stackURLs, void (^completionBlock)(int mipCounter, NSURL *destinationURL))
{
	NSOpenPanel *destinationPanel = [NSOpenPanel openPanel];
	// Could use the following to set the starting directory for the save panel:
	//	[spanel setDirectory:[path stringByExpandingTildeInPath]];
	destinationPanel.title = @"Save Generated MIPs In Folder...";
	destinationPanel.message = @"Choose the folder in which your MIP data will be saved";
	destinationPanel.prompt = @"Choose";
	destinationPanel.allowsMultipleSelection = FALSE;
	destinationPanel.canChooseDirectories = TRUE;
	destinationPanel.canChooseFiles = FALSE;
	destinationPanel.canCreateDirectories = TRUE;
	
	[destinationPanel beginWithCompletionHandler:^(NSInteger result)
	 {
		 dispatch_async(dispatch_get_main_queue(),
						^{
							if (result != NSOKButton)
								return;
							ProcessStacksIntoMIPsSavingAt(stackURLs, destinationPanel.URL, completionBlock);
						});
	 }];
}

void ProcessStacksIntoMIPs(void (^completionBlock)(int mipCounter, NSURL *destinationURL))
{
	NSOpenPanel *panel = [NSOpenPanel openPanel];
	panel.allowsMultipleSelection = TRUE;
	panel.canChooseDirectories = TRUE;
	panel.title = @"Choose a batch of stacks to process...";
	panel.message = @"Select stack folders - a MIP will be generated for each one.";
	
	[panel beginWithCompletionHandler:^(NSInteger result)
	 {
		 if (result == NSFileHandlingPanelOKButton)
			 ProcessTheseStacksIntoMIPs(panel.URLs, completionBlock);
	 }];
}
