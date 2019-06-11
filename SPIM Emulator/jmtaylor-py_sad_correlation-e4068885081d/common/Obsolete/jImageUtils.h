/*
 *  jImageUtils.h
 *  scatter
 *
 *  Created by Jonathan Taylor on 10/05/2010.
 *  Copyright 2010 Durham University. All rights reserved.
 *
 */

#ifndef __JIMAGEUTILS_H__
#define __JIMAGEUTILS_H__ 1

void LoadGWorldContentsFromPNG(GWorldPtr gWorld, const char *filenameFormat, ...);
void LoadGWorldContentsFromPNG(GWorldPtr gWorld, const char *filenameFormat, va_list argList);

void LoadGWorldContentsFromImageFile(GWorldPtr gWorld, const char *filenameFormat, ...);
void LoadGWorldContentsFromImageFile(GWorldPtr gWorld, const char *filenameFormat, va_list argList);
void SaveGWorldContentsAsImageFile(GWorldPtr gWorldToSave, const char *filenameFormat, ...);
void SaveGWorldContentsAsImageFile(GWorldPtr gWorldToSave, const char *filenameFormat, va_list argList);

void DrawGWorldInWindow(GWorldPtr theGWorld, WindowRef theWindow);
void SaveBufferAsGreyscalePNG(short *data, int depth, int width, int height, const char *filename);

#endif
