#ifndef _MSL_PATH2FSS_H
#define _MSL_PATH2FSS_H

	OSErr __path2fss(const char * pathName, FSSpecPtr spec);
	OSErr __path2fss_old(const char * pathName, FSSpecPtr spec);
	OSErr __msl_path2fsr(const char * pathName, FSRefPtr theRef);
	OSErr __msl_path2splitfsr(const char * pathName, FSRefPtr theParentRef, HFSUniStr255 * theNewName);
	TextEncoding __msl_get_system_encoding(void);
	void __msl_text2unicode(const short theLength, const char *theText, HFSUniStr255 *theUnicodeText);
	void __msl_unicode2text(const HFSUniStr255 *theUnicodeText, short *theLength, char *theText);
#endif
