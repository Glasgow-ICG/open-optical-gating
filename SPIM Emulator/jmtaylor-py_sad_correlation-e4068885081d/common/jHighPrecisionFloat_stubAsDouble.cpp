/*
 *	jHighPrecisionFloat_stubAsDouble.h
 *
 *  Copyright 2011-2015 Jonathan Taylor. All rights reserved.
 *
 *  Class representing a floating-point number, but potentially to higher (or lower) precision
 *	than supported by the ubiquitous 'double' type.
 *
 *	This implementation is just a wrapper around the 'double' type.
 *	Its purpose is just to help test the migration of existing code to the jHighPrecisionFloat type,
 *	but without actually adding any extra precision(!).
 *	A modern compiler should hopefully be able to compile this into code that has little or no overhead
 *	compared to the native 'double' type, but of course it should not normally be used for "production" code!
 */

#ifdef USE_JREAL

const long jHighPrecisionFloat::precision = 52;

void Print(jHighPrecisionFloat x, const char *suffix)
{
	x.Print(suffix);
}

int floor_int(const jHighPrecisionFloat &val) { return int(floor(val.doubleVal())); } 		// Calculate floor(val) and convert to int
bool is_nan(const jHighPrecisionFloat &val) { return val.doubleVal() != val.doubleVal(); }
jHighPrecisionFloat fabs(const jHighPrecisionFloat &val) { return AllowPrecisionLossOnParam(fabs(val.doubleVal())); }
jHighPrecisionFloat abs(const jHighPrecisionFloat &val) { return AllowPrecisionLossOnParam(fabs(val.doubleVal())); }
jHighPrecisionFloat pow(const jHighPrecisionFloat &val, const jHighPrecisionFloat &power) { return AllowPrecisionLossOnParam(pow(val.doubleVal(), power.doubleVal())); }
jHighPrecisionFloat exp(const jHighPrecisionFloat &val) { return AllowPrecisionLossOnParam(exp(val.doubleVal())); }
jHighPrecisionFloat sqrt(const jHighPrecisionFloat &val) { return AllowPrecisionLossOnParam(sqrt(val.doubleVal())); }
jHighPrecisionFloat log(const jHighPrecisionFloat &val) { return AllowPrecisionLossOnParam(log(val.doubleVal())); }
jHighPrecisionFloat sin(const jHighPrecisionFloat &val) { return AllowPrecisionLossOnParam(sin(val.doubleVal())); }
jHighPrecisionFloat sinh(const jHighPrecisionFloat &val) { return AllowPrecisionLossOnParam(sinh(val.doubleVal())); }
jHighPrecisionFloat cos(const jHighPrecisionFloat &val) { return AllowPrecisionLossOnParam(cos(val.doubleVal())); }
jHighPrecisionFloat cosh(const jHighPrecisionFloat &val) { return AllowPrecisionLossOnParam(cosh(val.doubleVal())); }
jHighPrecisionFloat tan(const jHighPrecisionFloat &val) { return AllowPrecisionLossOnParam(tan(val.doubleVal())); }
jHighPrecisionFloat asin(const jHighPrecisionFloat &val) { return AllowPrecisionLossOnParam(asin(val.doubleVal())); }
jHighPrecisionFloat acos(const jHighPrecisionFloat &val) { return AllowPrecisionLossOnParam(acos(val.doubleVal())); }
jHighPrecisionFloat atan2(const jHighPrecisionFloat &y, const jHighPrecisionFloat &x) { return AllowPrecisionLossOnParam(atan2(y.doubleVal(), x.doubleVal())); }
jHighPrecisionFloat sign(const jHighPrecisionFloat &val) { return AllowPrecisionLossOnParam(copysign(1.0, val.doubleVal())); }

jHighPrecisionFloat jHighPrecisionFloat::dbl_max(void) { return AllowPrecisionLossOnParam(DBL_MAX); }
jHighPrecisionFloat jHighPrecisionFloat::dbl_min(void) { return AllowPrecisionLossOnParam(DBL_MIN); }
jHighPrecisionFloat jHighPrecisionFloat::nan(void) { return AllowPrecisionLossOnParam(GSL_NAN); }
jHighPrecisionFloat jHighPrecisionFloat::lnpi(void) { return AllowPrecisionLossOnParam(M_LNPI); }
jHighPrecisionFloat jHighPrecisionFloat::ln2(void) { return AllowPrecisionLossOnParam(M_LN2); }
jHighPrecisionFloat jHighPrecisionFloat::epsilon(void) { return AllowPrecisionLossOnParam(GSL_DBL_EPSILON); }

jreal gsl_sf_lnpoch(const jreal &a, const jreal &x)
{
	return AllowPrecisionLossOnParam(gsl_sf_lnpoch(AllowPrecisionLossReadingValue(a), AllowPrecisionLossReadingValue(x)));
}
int gsl_sf_legendre_sphPlm_array(const int lmax, int m, const jreal &x, jreal * result_array)
{
	JHPF_TYPE resultDouble[lmax+1];
	int gslResult = gsl_sf_legendre_sphPlm_array(lmax, m, AllowPrecisionLossReadingValue(x), resultDouble);
	for (int i = 0; i <= lmax; i++)
		result_array[i] = AllowPrecisionLossOnParam(resultDouble[i]);
	return gslResult;
}

jreal gsl_sf_log_1plusx(const jreal &x)
{
	return AllowPrecisionLossOnParam(gsl_sf_log_1plusx(AllowPrecisionLossReadingValue(x)));
}

int gsl_sf_bessel_Jn_array(int nmin, int nmax, jreal &x, jreal *result_array)
{
	// TODO: when I implement this fully myself I should probably make it so that it handles underflow gracefully and silently.
	// I should then check everywhere I call this, because it looks like there are hacks in several different places!
	JHPF_TYPE resultDouble[nmax+1];
	int gslResult = gsl_sf_bessel_Jn_array(nmin, nmax, AllowPrecisionLossReadingValue(x), resultDouble);
	ALWAYS_ASSERT(nmin == 0);
	for (int i = 0; i <= nmax; i++)
		result_array[i] = AllowPrecisionLossOnParam(resultDouble[i]);
		return gslResult;
}

int gsl_sf_bessel_jl_array(const int lmax, const jreal &x, jreal *result_array)
{
	JHPF_TYPE resultDouble[lmax+1];
	int gslResult = gsl_sf_bessel_jl_array(lmax, AllowPrecisionLossReadingValue(x), resultDouble);
	for (int i = 0; i <= lmax; i++)
		result_array[i] = AllowPrecisionLossOnParam(resultDouble[i]);
	return gslResult;
}

jComplexR z_bessel_jl(const int l, const jComplexR &x)
{
	return AllowPrecisionLossOnParam(z_bessel_jl(l, AllowPrecisionLossReadingValue(x)));
}

jComplexVectorR z_bessel_jl_array(const int lmax, const jComplexR &z)
{
	jComplexVectorR resultComplexR(lmax+1);
	jComplexVector resultComplex = z_bessel_jl_array(lmax, AllowPrecisionLossReadingValue(z));
	for (int i = 0; i <= lmax; i++)
		resultComplexR[i] = AllowPrecisionLossOnParam(resultComplex[i]);
	return resultComplexR;
}

#endif
