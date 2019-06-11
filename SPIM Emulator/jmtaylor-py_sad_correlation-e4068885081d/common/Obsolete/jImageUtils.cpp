/*
 *  jImageUtils.cpp
 *  scatter
 *
 *  Created by Jonathan Taylor on 10/05/2010.
 *  Copyright 2010 Durham University. All rights reserved.
 *
 */

#if HAS_OS_X_GUI
	#include <QuickTime/QuickTime.h>
#endif
#include "jImageUtils.h"

#if 0
// These are obsolete. I don't think I actually call them from anywhere.
// If I need this in Common, see code in scatter/FieldWindow.mm
void SaveGWorldContentsAsImageFile(GWorldPtr gWorldToSave, const char *filenameFormat, ...)
{
	va_list		argList;
	va_start(argList, filenameFormat);
	SaveGWorldContentsAsImageFile(gWorldToSave, filenameFormat, argList);
}

void SaveBitmapAsImageFile(NSBitmapImageRep *imageToSave, const char *filenameFormat, va_list argList)
{
	// TODO: implement this in modern Cocoa, including determining required format from the file type suffix
	ALWAYS_ASSERT(0);
#if HAS_CARBON
	// Write the resultant image to a separate file
	OSStatus err;

	// find and open the PNG Graphics Exporter component
	GraphicsExportComponent exporter = 0;
	err = OpenADefaultComponent(GraphicsExporterComponentType,
								fileType,
								&exporter);
	ALWAYS_ASSERT_NOERR(err);

	// Set the input for the graphics exporter
	err = GraphicsExportSetInputGWorld(exporter, gWorldToSave);
	ALWAYS_ASSERT_NOERR(err);

	// Set the destination for the export    
	FSSpec outputFSSpec;            
	Str255 pString;
	pString[0] = snprintf((char *)pString + 1, 255, ":::");
	pString[0] += vsnprintf((char *)pString + pString[0] + 1, 255 - pString[0], filenameFormat, argList);
	printf("Output %s\n", (char*)(pString+1));
	err = FSMakeFSSpec(0, 0, pString, &outputFSSpec);
	if ((err != fnfErr) && (err != noErr))
		ALWAYS_ASSERT_NOERR(err);
	err = GraphicsExportSetOutputFile(exporter, &outputFSSpec);
	ALWAYS_ASSERT_NOERR(err);

	// the depth to use for the exported image
	err = GraphicsExportSetDepth(exporter, k24RGBPixelFormat);
	ALWAYS_ASSERT_NOERR(err);

	// do the export
	err = GraphicsExportDoExport(exporter, NULL);
	ALWAYS_ASSERT_NOERR(err);

	// Close the exporter component
	err = CloseComponent(exporter);
	ALWAYS_ASSERT_NOERR(err);
#else
	fprintf(stderr, "Asked to export PNG\n");
#endif
}

void LoadGWorldContentsFromImageFile(GWorldPtr gWorld, const char *filenameFormat, ...)
{
	va_list		argList;
	va_start(argList, filenameFormat);
	LoadGWorldContentsFromImageFile(gWorld, filenameFormat, argList);
}

void LoadGWorldContentsFromImageFile(GWorldPtr gWorld, const char *filenameFormat, va_list argList)
{
#if HAS_CARBON
	// Read an image from a PNG file
	OSStatus err;

	// find and open the PNG Graphics Importer component
	GraphicsImportComponent importer = 0;
	err = OpenADefaultComponent(GraphicsImporterComponentType,
								kQTFileTypePNG,
								&importer);
	ALWAYS_ASSERT_NOERR(err);

	// Set the input for the graphics exporter
	err = GraphicsImportSetGWorld(importer, gWorld, NULL);
	ALWAYS_ASSERT_NOERR(err);

	// Set the destination for the import
	FSSpec inputFSSpec;            
	Str255 pString;
	pString[0] = snprintf((char *)pString + 1, 255, ":::");
	pString[0] += vsnprintf((char *)pString + pString[0] + 1, 255 - pString[0], filenameFormat, argList);
	FSMakeFSSpec(0, 0, pString, &inputFSSpec);
	err = GraphicsImportSetDataFile(importer, &inputFSSpec);
	ALWAYS_ASSERT_NOERR(err);

	// do the import
	err = GraphicsImportDraw(importer);
	ALWAYS_ASSERT_NOERR(err);

	// Close the importer component
	err = CloseComponent(importer);
	ALWAYS_ASSERT_NOERR(err);
#else
	fprintf(stderr, "Asked to import PNG\n");
#endif
}

void DrawGWorldInWindow(GWorldPtr theGWorld, WindowRef theWindow)
{
#if HAS_CARBON
	CGrafPtr		aSavedPort;
	GDHandle		aSavedGDevice;
	Rect			gWorldBounds, windowBounds;

	GetGWorld(&aSavedPort, &aSavedGDevice);
	PixMapHandle pixMap = GetGWorldPixMap(theGWorld); 
	LockPixels(pixMap);
	
	SetPortWindowPort(theWindow);
	
	CopyBits( (BitMap *) *pixMap, (BitMap *) GetPortBitMapForCopyBits(GetWindowPort(theWindow)), GetPortBounds(theGWorld, &gWorldBounds), GetWindowPortBounds(theWindow, &windowBounds), srcCopy, NULL );   	// blit from frameGWorld1 to screen pixmap
	ALWAYS_ASSERT(QDError() == noErr);
	QDFlushPortBuffer(GetWindowPort(theWindow), NULL);
	SetGWorld(aSavedPort, aSavedGDevice);
#else
	fprintf(stderr, "Asked to DrawGWorldInWindow\n");
#endif
}

void SaveBufferAsGreyscalePNG(short *data, int depth, int width, int height, const char *filename)
{
#if HAS_CARBON
	GWorldPtr		theGWorld;
	Rect		theRect = { 0, 0, height, width };
	OSErr anErr = NewGWorld(&theGWorld, 32, &theRect, NULL, NULL, 0);
	ALWAYS_ASSERT_NOERR(anErr);
	PixMapHandle pixMap = GetGWorldPixMap(theGWorld);
	unsigned int *baseAddr = (unsigned int *)GetPixBaseAddr(pixMap);
	int rowSize = GetPixRowBytes(pixMap) / 4;
	
	ALWAYS_ASSERT(depth <= 15);
	for (int y = 0; y < height; y++)
		for (int x = 0; x < width; x++)
		{
			RGBColor rgb;
			rgb.red = rgb.green = rgb.blue = data[x + y * width] << (16 - depth);
			unsigned int pixVal = ((rgb.red & 0xFF00) << 0) | ((rgb.green & 0xFF00) << 8) | ((rgb.blue & 0xFF00) << 16);
			baseAddr[x + y*rowSize] = pixVal;
		}
		
	SaveGWorldContentsAsPNG(theGWorld, filename);
	
	DisposeGWorld(theGWorld);	
#else
	fprintf(stderr, "Asked to SaveBufferAsGreyscalePNG\n");
#endif
}
#endif
