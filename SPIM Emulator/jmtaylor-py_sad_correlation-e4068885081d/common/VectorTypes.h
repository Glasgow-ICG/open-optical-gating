/*
 *	VectorTypes.h
 *
 *	Copyright 2010-2015 Jonathan Taylor. All rights reserved.
 *
 *  Platform-independent definitions of basic CPU vector types
 *
 */

#ifndef __VECTOR_TYPES_H__
#define __VECTOR_TYPES_H__

#include "jOSMacros.h"

#if HAS_SSE
	#if __SSE3__
		#include <pmmintrin.h>
	#else
		#include <xmmintrin.h>
	#endif

	typedef __m128 vFloat;
	typedef __m128i vUChar;
	typedef __m128d vDouble;
#elif HAS_ALTIVEC
	#if __SPU__
		#include <spu_intrinsics.h>
		#include <spu_mfcio.h> /* constant declarations for the MFC */
		#include <simdmath.h>
	#else
		#ifndef __APPLE_ALTIVEC__
			#include <altivec.h>
	
			// altivec.h defines bool for its own purposes, but my existing code uses it
			// in its normal form all over the place. Undefine it to prevent compile errors!
			#undef bool
		#endif
	#endif
	
	typedef vector float vFloat;
	typedef vector unsigned int vUInt32;
	typedef vector unsigned char vUChar;
#endif

#if HAS_ALTIVEC || HAS_SSE
	typedef union
	{
		vFloat	vf;
	//	vUInt32	v32;
		vUChar	vc;
		float	f[4];
		long	l[4];
		short	s[8];
	} VecUnion;
#endif

#endif
