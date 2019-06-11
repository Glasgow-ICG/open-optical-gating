//
//  BoundsRect.h
//
//	Copyright 2014-2015 Jonathan Taylor. All rights reserved.
//
//	OS X specific code that allows us to insert markers into the Shark system trace
//	as an aid for performance analysis.
//

#ifndef __SIGNPOSTS_H__
#define __SIGNPOSTS_H__ 1

#if OS_X
	// Macros used to insert debug signposts into Shark system trace
	// While it would be nice to check the return codes, we get errors if Shark isn't running
	// Since I don't know how to check whether Shark is currently running, I'm just going to
	// ignore the return codes.
	#include <sys/syscall.h>
	#include <sys/kdebug.h>
	#define DEBUG_SIGNPOST4(CODE, P1, P2, P3, P4)	syscall(SYS_kdebug_trace, APPSDBG_CODE(DBG_MACH_CHUD, (CODE)) | DBG_FUNC_NONE, (P1), (P2), (P3), (P4))
	#define DEBUG_SIGNPOST(CODE)				DEBUG_SIGNPOST4((CODE), 0, 0, 0, 0)
	#define DEBUG_SIGNPOST1(CODE, P1)			DEBUG_SIGNPOST4((CODE), (P1), 0, 0, 0)
	#define DEBUG_SIGNPOST2(CODE, P1, P2)		DEBUG_SIGNPOST4((CODE), (P1), (P2), 0, 0)
	#define DEBUG_BEGIN_SIGNPOST(CODE, PARM)	syscall(SYS_kdebug_trace, APPSDBG_CODE(DBG_MACH_CHUD, (CODE)) | DBG_FUNC_START, (PARM), 0, 0, 0)
	#define DEBUG_BEGIN_SIGNPOST2(CODE, PARM, PARM2)	syscall(SYS_kdebug_trace, APPSDBG_CODE(DBG_MACH_CHUD, (CODE)) | DBG_FUNC_START, (PARM), (PARM2), 0, 0)
	#define DEBUG_END_SIGNPOST(CODE, PARM)		syscall(SYS_kdebug_trace, APPSDBG_CODE(DBG_MACH_CHUD, (CODE)) | DBG_FUNC_END, (PARM), 0, 0, 0)
#else
	#define DEBUG_SIGNPOST(CODE)				do { } while (0)
	#define DEBUG_SIGNPOST1(CODE, P1)			do { } while (0)
	#define DEBUG_SIGNPOST2(CODE, P1, P2)		do { } while (0)
	#define DEBUG_BEGIN_SIGNPOST(CODE, PARM)	do { } while (0)
	#define DEBUG_END_SIGNPOST(CODE, PARM)		do { } while (0)
#endif

#endif
