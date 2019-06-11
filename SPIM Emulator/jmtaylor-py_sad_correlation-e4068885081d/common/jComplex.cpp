/*
 *	jComplex.cpp
 *
 *  Copyright 2011-2015 Jonathan Taylor. All rights reserved.
 *
 *  A few utility functions for complex numbers.
 */

#ifdef USES_GSL
	#define COMPILE_JCOMPLEX_GSL_INTERFACE 1
#else
	#ifdef __GSL_COMPLEX_H__
		#define COMPILE_JCOMPLEX_GSL_INTERFACE 1
	#else
		#define COMPILE_JCOMPLEX_GSL_INTERFACE 0
	#endif
#endif

#include "jComplex.h"

void Print(jComplex z, const char *suffix)
{
	printf("{%.12le, %.12le}%s", z.real(), z.imag(), suffix);
}

#ifdef USE_JREAL
void Print(jComplexR z, const char *suffix)
{
//	printf("{%.12le, %.12le}", AllowPrecisionLossReadingValue(z.real()), AllowPrecisionLossReadingValue(z.imag()));
	printf("{");
	::Print(real(z), ",");
	::Print(imag(z), "}");
	printf("%s", suffix);
}
#endif

#if COMPILE_JCOMPLEX_GSL_INTERFACE
void Print(gsl_complex z, const char *suffix)
{
	printf("{%le, %le}%s", GSL_REAL(z), GSL_IMAG(z), suffix);
}

template<> jComplexAsStdBase<double>::jComplexAsStdBase(const gsl_complex &z) : complex<double>(GSL_REAL(z), GSL_IMAG(z))
{
}

#ifdef USE_JREAL
template<> jComplexAsStdBase<jreal>::jComplexAsStdBase(const gsl_complex &z) : complex<jreal>(jreal(GSL_REAL(z)), jreal(GSL_IMAG(z)))
{
}
#endif

#endif

#ifdef __JCOMPLEX_AS_VECTOR_H__
  #if COMPILE_JCOMPLEX_GSL_INTERFACE
	jComplexAsVector::jComplexAsVector(const gsl_complex &inZ)
	{
		SetReIm(GSL_REAL(inZ), GSL_IMAG(inZ));
	}
  #endif
#endif
