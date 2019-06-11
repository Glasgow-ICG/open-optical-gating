/*
 *	jHighPrecisionFloat.cpp
 *
 *  Copyright 2011-2015 Jonathan Taylor. All rights reserved.
 *
 *  Class representing a floating-point number, but potentially to higher (or lower) precision
 *	than supported by the ubiquitous 'double' type.
 */

#include "jHighPrecisionFloat.h"
#include "jComplex.h"

#if JREAL_DEFINED
jComplexR exp_i(jreal radianAngle)
{
	return jComplexR(cos(radianAngle), sin(radianAngle));
}

jComplexR exp_i(jComplexR z)
{
	// exp(i(a+ib)) = exp(ia) * exp(-b)
	return exp_i(z.real()) * exp(-z.imag());
}

double AllowPrecisionLossReadingValue(jreal val) { return val.doubleVal(); }
double AllowPrecisionLossReadingValue_mayAlreadyBeDouble(jreal val) { return val.doubleVal(); }
double AllowPrecisionLossReadingValue_mayAlreadyBeDouble(double val) { return val; }
jreal AllowPrecisionLossOnParam(double val) { return jreal(val, 0); }

#else

double AllowPrecisionLossReadingValue(jreal val) { return val; }
double AllowPrecisionLossReadingValue_mayAlreadyBeDouble(jreal val) { return val; }
jreal AllowPrecisionLossOnParam(double val) { return val; }

// (Disabling this next line because it should be covered by the Print(double) variant defined below
//void Print(jreal x, const char *suffix) { printf("%.17le%s", x, suffix); }

#endif

coord3R AllowPrecisionLossOnParam(coord3 val) { return coord3R(AllowPrecisionLossOnParam(val.x), AllowPrecisionLossOnParam(val.y), AllowPrecisionLossOnParam(val.z)); }
coordC3R AllowPrecisionLossOnParam(coordC3 val) { return coordC3R(AllowPrecisionLossOnParam(val.x), AllowPrecisionLossOnParam(val.y), AllowPrecisionLossOnParam(val.z)); }
jComplexR AllowPrecisionLossOnParam(jComplex val) { return jComplexR(AllowPrecisionLossOnParam(val.real()), AllowPrecisionLossOnParam(val.imag())); }
coord3 AllowPrecisionLossReadingValue(coord3R val) { return coord3(AllowPrecisionLossReadingValue(val.x), AllowPrecisionLossReadingValue(val.y), AllowPrecisionLossReadingValue(val.z)); }
coordC3 AllowPrecisionLossReadingValue(coordC3R val) { return coordC3(AllowPrecisionLossReadingValue(val.x), AllowPrecisionLossReadingValue(val.y), AllowPrecisionLossReadingValue(val.z)); }
jComplex AllowPrecisionLossReadingValue(jComplexR val) { return jComplex(AllowPrecisionLossReadingValue(val.real()), AllowPrecisionLossReadingValue(val.imag())); }

void Print(double x, const char *suffix)
{
    printf("%.16le%s", x, suffix);
}
