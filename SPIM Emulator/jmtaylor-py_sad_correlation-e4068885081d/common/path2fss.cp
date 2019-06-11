#include <Carbon/Carbon.h>
#include "path2fss.h"

#if __MACH__

#define _MSL_USE_NEW_FILE_APIS 1
#include <string.h>

/* function prototypes for externally defined functions */

extern char __msl_system_has_new_file_apis(void);


static void copy_c_to_pascal_str(StringPtr dst, const char * src)
{
	size_t len = strlen(src);
	
	memcpy(&dst[1], &src[0], len);
	
	dst[0] = len;
}

/* JWW - If only the old APIs are used, keep the old definition of __path2fss, but if
		the new APIs might be used, redefine it to __path2fss_old so the new APIs can
		have a chance to convert paths. */
#if _MSL_USE_OLD_FILE_APIS && (!(_MSL_USE_NEW_FILE_APIS))
OSErr __path2fss(const char * pathName, FSSpecPtr spec)
asd
#else
OSErr __path2fss_old(const char * pathName, FSSpecPtr spec)
#endif /* _MSL_USE_OLD_FILE_APIS && (!(_MSL_USE_NEW_FILE_APIS)) */
{
	Str255			pathNameStr;
	char			cPathNameStr[256];
	char *			fileName;
	int				fileNameLen;
	HVolumeParam	vpb;
	CInfoPBRec		cpb;
	OSErr			ioResult, result = noErr;
	
	if (!pathName || !*pathName || strlen(pathName) > 255)
		return(bdNamErr);

/* Extract file name (if any) */
	
	strcpy(cPathNameStr, pathName);
	
	fileName = strrchr(cPathNameStr, ':');
	
	if (!fileName++)
		fileName = cPathNameStr;
	
	fileNameLen = strlen(fileName);
	
/*	if (!fileNameLen || fileNameLen > sizeof(spec->name) - 1 || *fileName == '.')   /*- hh 9710258 -*/
/* Want to allow filenames begining with '.'.  */
	if (!fileNameLen || fileNameLen > (int)(sizeof(spec->name) - 1))
		return(bdNamErr);
	
	copy_c_to_pascal_str(spec->name, fileName);

/* Use PBGetCatInfo to get parID */
	
	copy_c_to_pascal_str(pathNameStr, pathName);
	
	cpb.hFileInfo.ioNamePtr = pathNameStr;
	cpb.hFileInfo.ioVRefNum = 0;
	cpb.hFileInfo.ioFDirIndex = 0;
	cpb.hFileInfo.ioDirID = 0;
	
	ioResult = PBGetCatInfoSync((CInfoPBPtr) &cpb);
	
	if (ioResult)
	{
		if (ioResult != fnfErr)
			return(ioResult);
	
	/* If file not found, we still need to get parID */
	/* Truncate path name and try again              */
		
		if (fileName == cPathNameStr)
			*fileName++ = ':';
		
		*fileName = 0;
		
		copy_c_to_pascal_str(pathNameStr, cPathNameStr);
		
		if ((ioResult = PBGetCatInfoSync((CInfoPBPtr) &cpb)) != 0)
			return(ioResult);
		
		if (!(cpb.hFileInfo.ioFlAttrib & ioDirMask))		/* make sure we got a directory */
			return(bdNamErr);
		
		spec->parID = cpb.dirInfo.ioDrDirID;
		
		result = fnfErr;
	}
	else
	{
		if (cpb.hFileInfo.ioFlAttrib & ioDirMask)				/* see if we got a directory */
			/*return(notAFileErr);*/
			result = notAFileErr;							/*- mm 980416 -*/
		
		spec->parID = cpb.hFileInfo.ioFlParID;
	}

/* Use PBHGetVInfo to get vRefNum */
	
	copy_c_to_pascal_str(pathNameStr, pathName);
	
	vpb.ioNamePtr = pathNameStr;
	vpb.ioVRefNum = 0;
	vpb.ioVolIndex = -1;
	
	ioResult = PBHGetVInfoSync((HParmBlkPtr) &vpb);
	
	if (ioResult)
		return(ioResult);
	
	spec->vRefNum = vpb.ioVRefNum;
	
	return(result);
}



#if _MSL_USE_NEW_FILE_APIS
char __msl_system_has_new_file_apis(void)
{
	static char hasNewAPIs = false;
	static char gestaltProbed = false;
	
	if (!gestaltProbed)
	{
		OSErr theErr;
		long theResponse;
		
		theErr = Gestalt(gestaltFSAttr, &theResponse);
		
		if ((theErr == noErr) && (theResponse & (1 << gestaltHasHFSPlusAPIs)))
			hasNewAPIs = true;
		
		gestaltProbed = true;
	}
	
	return hasNewAPIs;
}

OSErr __path2fss(const char * pathName, FSSpecPtr spec)
{
#if _MSL_USE_OLD_AND_NEW_FILE_APIS
	if (__msl_system_has_new_file_apis())
	{
#endif /* _MSL_USE_OLD_AND_NEW_FILE_APIS */
#if _MSL_USE_NEW_FILE_APIS
		OSErr theErr;
		FSRef theRef;
		FSCatalogInfo theInfo;
		HFSUniStr255 theName;
		Str255 thePName;
		
		if (!pathName || !*pathName || strlen(pathName) > 255)
			return (bdNamErr);
		
		theErr = __msl_path2fsr(pathName, &theRef);
		
		if (theErr == noErr)
			theErr = FSGetCatalogInfo(&theRef, kFSCatInfoNone, NULL, NULL, spec, NULL);
		else if (theErr == fnfErr)
		{
			theErr = __msl_path2splitfsr(pathName, &theRef, &theName);
			
			if (theErr == noErr)
			{
				theErr = FSGetCatalogInfo(&theRef, kFSCatInfoVolume + kFSCatInfoNodeID, &theInfo,
					NULL, NULL, NULL);
				
				if (theErr == noErr)
				{
					copy_c_to_pascal_str(thePName, pathName);
					theErr = FSMakeFSSpec(theInfo.volume, theInfo.nodeID, thePName, spec);
				}
			}
		}
		
		return theErr;
#endif /* _MSL_USE_NEW_FILE_APIS */
#if _MSL_USE_OLD_AND_NEW_FILE_APIS
	}
	else
	{
#endif /* _MSL_USE_OLD_AND_NEW_FILE_APIS */
#if _MSL_USE_OLD_FILE_APIS
		return __path2fss_old(pathName, spec);
#endif /* _MSL_USE_OLD_FILE_APIS */
#if _MSL_USE_OLD_AND_NEW_FILE_APIS
	}
#endif /* _MSL_USE_OLD_AND_NEW_FILE_APIS */
}

/* Find the text encoding currently in use by the system */
TextEncoding __msl_get_system_encoding(void)
{
	OSStatus theStatus;
	TextEncoding theEncoding;
	
	theStatus = UpgradeScriptInfoToTextEncoding(smSystemScript, kTextLanguageDontCare,
		kTextRegionDontCare, NULL, &theEncoding);
	
	if (theStatus != noErr)
		theEncoding = kTextEncodingMacRoman;
	
	return theEncoding;
}

/* Turn a pathname into a FSRef */
OSErr __msl_path2fsr(const char * pathName, FSRefPtr theRef)
{
	OSErr			theErr;
	Boolean			isFolder;
	Boolean			wasAlias;
	FSRef			theParentRef;
	HFSUniStr255	theName;
	
	theErr = __msl_path2splitfsr(pathName, &theParentRef, &theName);
	
	if (theErr == noErr)
		theErr = FSMakeFSRefUnicode(&theParentRef, theName.length, theName.unicode,
			__msl_get_system_encoding(), theRef);
	
	if ((theErr == noErr) && ((UInt32) FSResolveAliasFileWithMountFlags != 0))
		theErr = FSResolveAliasFileWithMountFlags(theRef, true, &isFolder, &wasAlias,
			kResolveAliasFileNoUI);
	
	return theErr;
}

/* Convert a C string to a unicode HFSUniStr255 */
void __msl_text2unicode(const short theLength, const char *theText, HFSUniStr255 *theUnicodeText)
{
	int i;
	OSStatus theStatus;
	ByteCount theSourceUsed;
	ByteCount theConvertedLength;
	TextToUnicodeInfo theConverterInfo;
	
	theStatus = CreateTextToUnicodeInfoByEncoding(__msl_get_system_encoding(), &theConverterInfo);
	
	if (theStatus == noErr)
	{
		theStatus = ConvertFromTextToUnicode(theConverterInfo, theLength, theText, kNilOptions,
			0, NULL, NULL, NULL, sizeof(theUnicodeText->unicode), &theSourceUsed, &theConvertedLength,
			(UniCharArrayPtr) &(theUnicodeText->unicode));
		
		theUnicodeText->length = theConvertedLength / 2;
		
		DisposeTextToUnicodeInfo(&theConverterInfo);
	}
	
	if (theStatus != noErr)
	{
		theUnicodeText->length = theLength;
		
		for (i = 0; i < theLength; i++)
			theUnicodeText->unicode[i] = theText[i];
	}
}

/* Convert a unicode HFSUniStr255 to a C string */
void __msl_unicode2text(const HFSUniStr255 *theUnicodeText, short *theLength, char *theText)
{
	int i;
	OSStatus theStatus;
	ByteCount theSourceUsed;
	ByteCount theConvertedLength;
	UnicodeToTextInfo theConverterInfo;
	
	theStatus = CreateUnicodeToTextInfoByEncoding(__msl_get_system_encoding(), &theConverterInfo);
	
	if (theStatus == noErr)
	{
		theStatus = ConvertFromUnicodeToText(theConverterInfo, theUnicodeText->length * 2,
			theUnicodeText->unicode, kUnicodeLooseMappingsMask, 0, NULL, NULL, NULL, *theLength,
			&theSourceUsed, &theConvertedLength, theText);
		
		*theLength = theSourceUsed / 2;
		
		DisposeUnicodeToTextInfo(&theConverterInfo);
	}
	
	if (theStatus != noErr)
	{
		*theLength = theUnicodeText->length;
		
		for (i = 0; i < *theLength; i++)
			theText[i] = theUnicodeText->unicode[i];
	}
}

/* Turn a pathname into a parent FSRef and a filename */
OSErr __msl_path2splitfsr(const char * pathName, FSRefPtr theParentRef, HFSUniStr255 * theNewName)
{
	int i;
	int position;
	int lastPosition = 0;		// JMT: initialize to stop gcc warning
	int nextPosition;
	int pathNameLength;
	char separatorFound;
	Boolean isFolder;
	Boolean wasAlias;
	FSSpec theSpec;
	FSRef theRef;
	OSErr theErr;
	
	pathNameLength = strlen(pathName);
	
	if (pathName[0] == ':')
	{
		theErr = HGetVol(NULL, &theSpec.vRefNum, &theSpec.parID);
		
		if (theErr != noErr)
		{
			theSpec.vRefNum = 0;
			theSpec.parID = 0;
		}
		
		theErr = FSMakeFSSpec(theSpec.vRefNum, theSpec.parID, "\p", &theSpec);
		
		if (theErr == noErr)
			theErr = FSpMakeFSRef(&theSpec, theParentRef);
		
		lastPosition = 1;
		separatorFound = true;
		
		while ((theErr == noErr) && (pathName[lastPosition] == ':'))
		{
			theRef = *theParentRef;
			theErr = FSGetCatalogInfo(&theRef, kFSCatInfoNone, NULL, NULL, NULL, theParentRef);
			lastPosition++;
		}
	}
	else
	{
		position = 0;
		separatorFound = false;
		while ((!separatorFound) && (position < pathNameLength))
		{
			if (pathName[position] == ':')
				separatorFound = true;
			else
				position++;
		}
		
		if (separatorFound)
		{
			theSpec.vRefNum = 0;
			theSpec.parID = 0;
			for (i = 0; i < position + 1; i++)
				theSpec.name[i + 1] = pathName[i];
			theSpec.name[0] = position + 1;
			theErr = FSpMakeFSRef(&theSpec, theParentRef);
			
			lastPosition = position + 1;
		}
		else
		{
			theErr = HGetVol(NULL, &theSpec.vRefNum, &theSpec.parID);
			
			if (theErr != noErr)
			{
				theSpec.vRefNum = 0;
				theSpec.parID = 0;
			}
			
			theErr = FSMakeFSSpec(theSpec.vRefNum, theSpec.parID, "\p", &theSpec);
			
			if (theErr == noErr)
				theErr = FSpMakeFSRef(&theSpec, theParentRef);
			
			__msl_text2unicode(pathNameLength, pathName, theNewName);
		}
	}
	
	while ((theErr == noErr) && separatorFound)
	{
		position = lastPosition;
		separatorFound = false;
		while ((!separatorFound) && (position < pathNameLength))
		{
			if (pathName[position] == ':')
				separatorFound = true;
			else
				position++;
		}
		
		nextPosition = position;
		
		__msl_text2unicode(nextPosition - lastPosition, &(pathName[lastPosition]), theNewName);
		
		if (separatorFound)
		{
			lastPosition = nextPosition + 1;
			
			if (pathName[lastPosition] == 0)
				separatorFound = false;
			else
			{
				if (theNewName->length == 0) /* Found a backup directory double colon :: */
					theErr = FSGetCatalogInfo(theParentRef, kFSCatInfoNone, NULL, NULL, NULL,
						theParentRef);
				else
				{
					theErr = FSMakeFSRefUnicode(theParentRef, theNewName->length,
						theNewName->unicode, __msl_get_system_encoding(), theParentRef);
					
					if ((theErr == noErr) && ((UInt32) FSResolveAliasFileWithMountFlags != 0))
						theErr = FSResolveAliasFileWithMountFlags(theParentRef, true, &isFolder,
							&wasAlias, kResolveAliasFileNoUI);
				}
			}
		}
	}
	
	return theErr;
}

#endif /* _MSL_USE_NEW_FILE_APIS */

#endif /* _MSL_CARBON_FILE_APIS */

/* Change record:
 * JFH 950816 First code release.
 * hh  971025 File names beginning with '.' can now be opened
 * mm  980416 Allow the selection of a directory and make a note of it.    MW00456
 * JWW 001030 Added routines to convert from pathnames to FSRef objects
 * JWW 010510 Added __msl_text2unicode as an extern helper instead of static just for path2fss.c
 * JWW 010529 Added __msl_unicode2text
 * JWW 010614 Added __msl_get_system_encoding and changed unicode routines to rely on it
 * JWW 010723 Made __msl_path2splitfsr more efficient and use HGetVol to get vRefNum and parID (thanks to Richard Buckle)
 * JWW 010803 Normalize FSSpecs with FSMakeFSSpec so things work in Classic on OS X
 * JWW 011008 Only use HGetVol when given a relative path, not an absolute path
 * JWW 011008 Allow relative pathnames to go "backwards" with :: when using new HFS+ APIs
 * JWW 011027 Added case for Mach-O
 * JWW 011126 Don't do anything when not using Carbon APIs (test _MSL_CARBON_FILE_APIS)
 * JWW 020430 Resolve aliases in pathname items when resolving pathnames
 * JWW 020515 Use kUnicodeLooseMappingsMask when converting from unicode to text
*/
