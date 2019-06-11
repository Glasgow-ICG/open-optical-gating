/*
 *	jComplexPair.h
 *
 *  Copyright 2011-2015 Jonathan Taylor. All rights reserved.
 *
 *  Class representing two complex numbers.
 *	There are in fact several variants, with different underlying storage formats.
 *	The vector-based implementations are the fastest.
 */

#ifndef __JCOMPLEX_PAIR_H__
#define __JCOMPLEX_PAIR_H__

#include "jComplex.h"
#include "VectorTypes.h"
#include "jComplexPairSplit.h"

typedef jComplexPairSplitT<jComplexR, jreal> jComplexPairSplit;

#ifdef USE_JREAL
	#define COMPLEX_PAIR_IS_VECTOR 0
	typedef jComplexPairSplit jComplexPair;
#else
	#if HAS_SSE
		#define COMPLEX_PAIR_IS_VECTOR 1
	  #if HAS_AVX
		#include "jComplexPairAsVector256.h"
		typedef jComplexPairAsVector256 jComplexPair;
	  #else
		#include "jComplexPairAsVector.h"
		typedef jComplexPairAsVector jComplexPair;
	  #endif
	#else
		#define COMPLEX_PAIR_IS_VECTOR 0
		typedef jComplexPairSplit<jComplex, double> jComplexPair;
	#endif
#endif

#endif
