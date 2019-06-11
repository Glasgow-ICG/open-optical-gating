/*
 *	jHighPrecisionFloat_mpfr.cpp
 *
 *  Copyright 2011-2015 Jonathan Taylor. All rights reserved.
 *
 *  Class representing a floating-point number, but potentially to higher (or lower) precision
 *	than supported by the ubiquitous 'double' type.
 *
 *	This implementation relies on the MPFR library to implement the underlying high-precision arithmetic.
 */

#include "jHighPrecisionFloat_mpfr.h"

const mpfr_rnd_t jHighPrecisionFloat_mpfr::rounding = MPFR_RNDN;
/*  Define the precision. It would be nice to be able to switch between precisions on the fly,
    but this is not trivial given the way I have set my code up. I have included assertions
    checking that we assign a variable with the same precision it had previously.
    This is violated when we call SetNMax after changing the precision - lookup tables have
    previously been initialized with a different precision.
    I can't see an immediate easy but safe solution to this, so for now I will leave this variable
    as a compile-time constant  */
const mpfr_prec_t jHighPrecisionFloat_mpfr::precision = 128;

bool IsExactIntegerValueWithZeroError(const jHighPrecisionFloat_mpfr &a)
{
	if (a.doubleErr() != 0)
		return false;
	double doubleVal = a.doubleVal();
	// Check that this is an integer value.
	// We cannot have complete confidence that this is the case if the value is large
	// (this is just a sloppy condition for now, though...)
	// Note that this is not a definitive test, but will at least pick up small integers
	if ((doubleVal == round(doubleVal)) &&
		(fabs(doubleVal) < 1e5))
	{
		return true;
	}
	return false;
}

bool IsExactIntegerValueWithZeroError(const jHighPrecisionFloat_mpfr &a, const jHighPrecisionFloat_mpfr &b)
{
	return IsExactIntegerValueWithZeroError(a) && IsExactIntegerValueWithZeroError(b);
}

jHighPrecisionFloat_mpfr::jHighPrecisionFloat_mpfr()
{
	mpfr_init2(__val, precision);
	mpfr_set_d(__val, 0.0, rounding);
	__err = 0;
}

jHighPrecisionFloat_mpfr::jHighPrecisionFloat_mpfr(const jHighPrecisionFloat_mpfr &obj)
{
	mpfr_init2(__val, precision);
	mpfr_set(__val, obj.mpfrVal(), rounding);
	__err = obj.doubleErr();
}

jHighPrecisionFloat_mpfr::jHighPrecisionFloat_mpfr(const char *numString, double inErr)
{
	mpfr_init2(__val, precision);
	mpfr_set_str(__val, numString, 10, rounding);
	__err = inErr;
}

jHighPrecisionFloat_mpfr::jHighPrecisionFloat_mpfr(int (*func)(mpfr_ptr, mpfr_rnd_t))
{
	mpfr_init2(__val, precision);
	func(__val, jHighPrecisionFloat_mpfr::rounding);
	__err = RoundingError(doubleVal(), 0.0);
}

jHighPrecisionFloat_mpfr::jHighPrecisionFloat_mpfr(int (*func)(mpfr_ptr, mpfr_srcptr, mpfr_rnd_t), const jHighPrecisionFloat_mpfr &srcVal, double inErr, bool noRoundingWillOccur)
{
	mpfr_init2(__val, precision);
	func(__val, srcVal.__val, jHighPrecisionFloat_mpfr::rounding);
	if (noRoundingWillOccur)
		__err = 0;
	else if (srcVal.doubleErr() == 0)
		__err = RoundingError(doubleVal(), 0.0);
	else
		__err = RoundingError(doubleVal(), inErr);
	if (__err != __err)
		printf("srcVal.doubleVal is %lf\n", srcVal.doubleVal());
	ALWAYS_ASSERT(__err == __err);
}

jHighPrecisionFloat_mpfr::jHighPrecisionFloat_mpfr(int (*func)(mpfr_ptr, mpfr_srcptr, mpfr_srcptr, mpfr_rnd_t), const jHighPrecisionFloat_mpfr &x, const jHighPrecisionFloat_mpfr &y, double inErr, bool noRoundingWillOccur)
{
	mpfr_init2(__val, precision);
	func(__val, x.__val, y.__val, jHighPrecisionFloat_mpfr::rounding);
	if (noRoundingWillOccur)
		__err = inErr;
	else
		__err = RoundingError(doubleVal(), inErr);
}

jHighPrecisionFloat_mpfr::jHighPrecisionFloat_mpfr(mpfr_ptr inVal, double inErr)
{
	// Although somewhat inefficient, this is sometimes necessary as a constructor
	mpfr_init2(__val, precision);
	mpfr_set(__val, inVal, rounding);
	__err = inErr;
}

jHighPrecisionFloat_mpfr::jHighPrecisionFloat_mpfr(double inVal)
{
	// If we are provided with a double value then we set that.
	// If it looks like an integer or a fraction up to 1/8, we assign it an error of 0
	// Otherwise we assume that truncation has occurred.
	mpfr_init2(__val, precision);
	mpfr_set_d(__val, inVal, rounding);
	if (round(inVal * 8) == inVal * 8)
		__err = 0;
	else
		__err = inVal * GSL_DBL_EPSILON;
}

jHighPrecisionFloat_mpfr::jHighPrecisionFloat_mpfr(double inVal, double inErr)
{
	// Explicit constructor with double value and associated error
	// Main use is to force an error of 0 for cases where we are happy with loss of precision
	// (e.g. sphere position is considered to be *exactly* the double-represented value)
	mpfr_init2(__val, precision);
	mpfr_set_d(__val, inVal, rounding);
	__err = inErr;
}

jHighPrecisionFloat_mpfr::jHighPrecisionFloat_mpfr(int inVal)
{
	mpfr_init2(__val, precision);
	mpfr_set_si(__val, inVal, rounding);
	__err = 0;
}

jHighPrecisionFloat_mpfr::jHighPrecisionFloat_mpfr(long inVal)
{
	mpfr_init2(__val, precision);
	mpfr_set_si(__val, inVal, rounding);
	__err = 0;
}

jHighPrecisionFloat_mpfr::~jHighPrecisionFloat_mpfr()
{
	mpfr_clear(__val);
}

void Print(jHighPrecisionFloat_mpfr x, const char *suffix)
{
	x.Print(suffix);
}

jHighPrecisionFloat_mpfr &jHighPrecisionFloat_mpfr::operator =(const jHighPrecisionFloat_mpfr &copy)
{
	// Do not call mpfr_init2, because this is not a constructor!
	// We have already been constructed, we are just being asked to set to a different value
	ALWAYS_ASSERT(mpfr_get_prec(__val) == mpfr_get_prec(copy.mpfrVal()));
	mpfr_set(__val, copy.mpfrVal(), rounding);
	__err = copy.doubleErr();
	return *this;
}

jHighPrecisionFloat_mpfr& jHighPrecisionFloat_mpfr::operator += (const jHighPrecisionFloat_mpfr &n)
{
	mpfr_add(__val, __val, n.mpfrVal(), rounding);
	if (IsExactIntegerValueWithZeroError(*this, n))		// Would need to be more careful if one integer was enormous...
		__err = 0;
	else
	{
		__err = sqrt(SQUARE(__err) + SQUARE(n.doubleErr()));
		__err = RoundingError(doubleVal(), __err);
	}
	return *this;
}

jHighPrecisionFloat_mpfr& jHighPrecisionFloat_mpfr::operator -= (const jHighPrecisionFloat_mpfr &n)
{
	mpfr_sub(__val, __val, n.mpfrVal(), rounding);
	if (IsExactIntegerValueWithZeroError(*this, n))		// Would need to be more careful if one integer was enormous...
		__err = 0;
	else
	{
		__err = sqrt(SQUARE(__err) + SQUARE(n.doubleErr()));
		__err = RoundingError(doubleVal(), __err);
	}
	return *this;
}

jHighPrecisionFloat_mpfr& jHighPrecisionFloat_mpfr::operator *= (const jHighPrecisionFloat_mpfr &n)
{
	double oldVal = doubleVal();
	double oldErr = doubleErr();
	mpfr_mul(__val, __val, n.mpfrVal(), rounding);
	double nVal = n.doubleVal();
	if (IsExactIntegerValueWithZeroError(*this, n))
		__err = 0;
	else
		__err = RoundingError(doubleVal(), sqrt(SQUARE(nVal * oldErr) + SQUARE(oldVal * n.doubleErr())));
	return *this;
}

jHighPrecisionFloat_mpfr& jHighPrecisionFloat_mpfr::operator /= (const jHighPrecisionFloat_mpfr &n)
{
	double oldVal = doubleVal();
	double oldErr = doubleErr();
	mpfr_div(__val, __val, n.mpfrVal(), rounding);
	if (IsExactIntegerValueWithZeroError(*this /*new value*/, n))
		__err = 0;
	else
		__err = RoundingError(doubleVal(), sqrt(SQUARE(oldErr) + SQUARE(oldVal * n.doubleErr() / n.doubleVal()))) / n.doubleVal();
	return *this;
}

int floor_int(const jHighPrecisionFloat_mpfr &val) { return int(floor(val.doubleVal())); } 		// Calculate floor(val) and convert to int
bool is_nan(const jHighPrecisionFloat_mpfr &val) { return val.doubleVal() != val.doubleVal(); }
jHighPrecisionFloat_mpfr fabs(const jHighPrecisionFloat_mpfr &val) { return jHighPrecisionFloat_mpfr(mpfr_abs, val, val.doubleErr(), false); }
jHighPrecisionFloat_mpfr abs(const jHighPrecisionFloat_mpfr &val) { return fabs(val); }
jHighPrecisionFloat_mpfr pow(const jHighPrecisionFloat_mpfr &val, double power)
{
	return jHighPrecisionFloat_mpfr(mpfr_pow, val, jHighPrecisionFloat_mpfr(power), val.doubleErr() * power * pow(val.doubleVal(), power-1), false);
}
jHighPrecisionFloat_mpfr exp(const jHighPrecisionFloat_mpfr &val) { return jHighPrecisionFloat_mpfr(mpfr_exp, val, val.doubleErr() * fabs(exp(val.doubleVal())), false); }
jHighPrecisionFloat_mpfr log(const jHighPrecisionFloat_mpfr &val) { return jHighPrecisionFloat_mpfr(mpfr_log, val, val.doubleErr() / fabs(val.doubleVal()), false); }
jHighPrecisionFloat_mpfr pow(const jHighPrecisionFloat_mpfr &val, const jHighPrecisionFloat_mpfr &power)
{
	double err = val.doubleErr() * fabs(power.doubleVal() * pow(val.doubleVal(), power.doubleVal() - 1));
	if ((IsExactIntegerValueWithZeroError(val)) &&
		(fabs(val.doubleVal()) == 1))
	{
		err = 0;
	}
	return jHighPrecisionFloat_mpfr(mpfr_pow, val, power, err, false);
}
jHighPrecisionFloat_mpfr sqrt(const jHighPrecisionFloat_mpfr &val)
{
	double err = val.doubleErr() * 0.5 / sqrt(val.doubleVal());
	if ((IsExactIntegerValueWithZeroError(val)) &&
		(val.doubleVal() == 1))
	{
		err = 0;
	}
	return jHighPrecisionFloat_mpfr(mpfr_sqrt, val, err, (err == 0.0));
}
jHighPrecisionFloat_mpfr sin(const jHighPrecisionFloat_mpfr &val) { return jHighPrecisionFloat_mpfr(mpfr_sin, val, fabs(cos(val.doubleVal())) * val.doubleErr(), false); }
jHighPrecisionFloat_mpfr sinh(const jHighPrecisionFloat_mpfr &val) { return jHighPrecisionFloat_mpfr(mpfr_sinh, val, val.doubleErr() * fabs(cosh(val.doubleVal())), false); }
//jHighPrecisionFloat_mpfr cos(const jHighPrecisionFloat_mpfr &val) { return jHighPrecisionFloat_mpfr(mpfr_cos, val, fabs(sin(val.doubleVal())) * val.doubleErr()); }
jHighPrecisionFloat_mpfr cos(const jHighPrecisionFloat_mpfr &val)
{
	return jHighPrecisionFloat_mpfr(mpfr_cos, val, fabs(sin(val.doubleVal())) * val.doubleErr(), false);
}
jHighPrecisionFloat_mpfr cosh(const jHighPrecisionFloat_mpfr &val) { return jHighPrecisionFloat_mpfr(mpfr_cosh, val, val.doubleErr() * fabs(sinh(val.doubleVal())), false); }
jHighPrecisionFloat_mpfr tan(const jHighPrecisionFloat_mpfr &val) { return jHighPrecisionFloat_mpfr(mpfr_tan, val, val.doubleErr() / fabs(SQUARE(cos(val.doubleVal()))), false); }
jHighPrecisionFloat_mpfr asin(const jHighPrecisionFloat_mpfr &val)
{
	double doubleVal = val.doubleVal();
	double doubleErr;
	if (fabs(doubleVal) == 1.0)
	{
		// These asserts are not guaranteed, but if our values are precisely ±1 then unless there
		// has been a staggering coincidence we should expect the errors to be zero (values initialized as exactly zero)
		// If these asserts are failed then we have a problem - the error diverges at this point
		ALWAYS_ASSERT(val.doubleErr() == 0);
		doubleErr = 0.0;
	}
	else
		doubleErr = val.doubleErr() / sqrt(1 - SQUARE(val.doubleVal()));
	return jHighPrecisionFloat_mpfr(mpfr_asin, val, doubleErr, (doubleErr == 0.0));
}
jHighPrecisionFloat_mpfr acos(const jHighPrecisionFloat_mpfr &val)
{
	double doubleVal = val.doubleVal();
	double doubleErr;
	if (fabs(doubleVal) == 1.0)
	{
		/*  This is a difficult scenario to handle. If our values are precisely ±1 but there is a genuine uncertainty
            on them, then we have a problem - the error diverges at this point.
            We can encounter this in practice though - e.g. when calculating a spherical translation vector between
            two spheres on the z axis.
            Although this solution is a little unsatisfactory, I am going to assume that (barring a staggering coincidence!)
            if the value comes out as exactly ±1 then the error is 0    */
		doubleErr = 0.0;
	}
	else
		doubleErr = val.doubleErr() / sqrt(1 - SQUARE(val.doubleVal()));
	return jHighPrecisionFloat_mpfr(mpfr_acos, val, doubleErr, (doubleErr == 0.0));
}
jHighPrecisionFloat_mpfr atan2(const jHighPrecisionFloat_mpfr &y, const jHighPrecisionFloat_mpfr &x)
{
	double yDbl = y.doubleVal(), xDbl = x.doubleVal();
#if 0
	double yOverXErr;
	if ((yDbl == 0.0) && (xDbl == 0.0))
	{
		// These asserts are not guaranteed, but if our values are precisely zero then unless there
		// has been a staggering coincidence we should expect the errors to be zero (values initialized as exactly zero)
		// If these asserts are failed then we have a problem - if there is uncertainty on y or x then the result
		// of atan2 could be anything!
		ALWAYS_ASSERT(y.doubleErr() == 0);
		ALWAYS_ASSERT(x.doubleErr() == 0);
		yOverXErr = 0.0;
	}
	else
	{
		yOverXErr = 1.0 / xDbl * sqrt(SQUARE(y.doubleErr()) + SQUARE(yDbl * x.doubleErr() / xDbl));
		yOverXErr = jHighPrecisionFloat_mpfr::RoundingError(yDbl/xDbl, yOverXErr);
	}
	// This is wrong
	double err = yOverXErr / (1 + SQUARE(atan2(yDbl, xDbl)));
#else
	double err;
	bool noRoundingWillOccur = false;
	if ((yDbl == 0.0) && (xDbl == 0.0))
	{
		// These asserts are not guaranteed, but if our values are precisely zero then unless there
		// has been a staggering coincidence we should expect the errors to be zero (values initialized as exactly zero)
		// If these asserts are failed then we have a problem - if there is uncertainty on y or x then the result
		// of atan2 could be anything!
		ALWAYS_ASSERT(y.doubleErr() == 0);
		ALWAYS_ASSERT(x.doubleErr() == 0);
		err = 0.0;
		noRoundingWillOccur = true;
	}
	else
		err = sqrt(SQUARE(xDbl * y.doubleErr()) + SQUARE(yDbl * x.doubleErr())) / (SQUARE(xDbl) + SQUARE(yDbl));
#endif
	return jHighPrecisionFloat_mpfr(mpfr_atan2, y, x, err, noRoundingWillOccur);
}

jHighPrecisionFloat_mpfr besselJn(int n, const jHighPrecisionFloat_mpfr &val)
{
	mpfr_t result;
	mpfr_init2(result, jHighPrecisionFloat_mpfr::precision);
	// Yes, mpfr_jn does indeed give Jn(!)
	mpfr_jn(result, n, val.mpfrVal(), jHighPrecisionFloat_mpfr::rounding);
	// Derivative is (J_{n-1} - J_{n+1}) / 2
	double d = val.doubleVal();
	double deriv = (gsl_sf_bessel_Jn(n-1, d) - gsl_sf_bessel_Jn(n+1, d)) / 2;
	double resultDouble = mpfr_get_d(result, jHighPrecisionFloat_mpfr::rounding);
	jHighPrecisionFloat_mpfr result2 = jHighPrecisionFloat_mpfr(result, jHighPrecisionFloat_mpfr::RoundingError(resultDouble, val.doubleErr() * deriv));
	mpfr_clear(result);
	return result2;
}

jHighPrecisionFloat_mpfr ln_gamma_function(const jHighPrecisionFloat_mpfr &val)
{
	// Derivative of ln(gamma) is (gamma' / gamma) which is the digamma (or psi) function
	double err = val.doubleErr() * gsl_sf_psi(val.doubleVal());
	return jHighPrecisionFloat_mpfr(mpfr_lngamma, val, err, false);
}

jHighPrecisionFloat_mpfr hypot(const jHighPrecisionFloat_mpfr &x, const jHighPrecisionFloat_mpfr &y)
{
	// We could do this using mpfr_hypot, but I'm just going to do it using the naive method
	// so that I don't have to faff with the error propagation!!
	return sqrt(SQUARE(x) + SQUARE(y));
}

jHighPrecisionFloat_mpfr sign(const jHighPrecisionFloat_mpfr &val)
{
	// Must return -1.0 when sign bit is set, or otherwise +1.0
	if (mpfr_signbit(val.mpfrVal()))
		return jHighPrecisionFloat_mpfr(-1);
	else
		return jHighPrecisionFloat_mpfr(+1);
}

// dbl_max and dbl_min are only used in a couple of very specific places in my code
// For now I'm going to ignore the problem and just return Nan
jHighPrecisionFloat_mpfr jHighPrecisionFloat_mpfr::dbl_max(void) { return jHighPrecisionFloat_mpfr::nan(); }
jHighPrecisionFloat_mpfr jHighPrecisionFloat_mpfr::dbl_min(void) { return jHighPrecisionFloat_mpfr::nan(); }
jHighPrecisionFloat_mpfr jHighPrecisionFloat_mpfr::nan(void) { jHighPrecisionFloat_mpfr result; mpfr_set_nan(result.mpfrVal_nonConst()); return result; }
jHighPrecisionFloat_mpfr jHighPrecisionFloat_mpfr::posinf(void) { jHighPrecisionFloat_mpfr result; mpfr_set_inf(result.mpfrVal_nonConst(), jHighPrecisionFloat_mpfr::precision); return result; }
jHighPrecisionFloat_mpfr jHighPrecisionFloat_mpfr::pi(void) { return jHighPrecisionFloat_mpfr(mpfr_const_pi); }

jHighPrecisionFloat_mpfr const_ln_pi = log(jHighPrecisionFloat_mpfr::pi());
jHighPrecisionFloat_mpfr const_epsilon = pow(jHighPrecisionFloat_mpfr(2), -jHighPrecisionFloat_mpfr::precision);		// TODO: need to check if this is correct or if it is ever so slightly too small
double const_epsilon_d = pow(2, -jHighPrecisionFloat_mpfr::precision);		// TODO: need to check if this is correct or if it is ever so slightly too small

jHighPrecisionFloat_mpfr jHighPrecisionFloat_mpfr::lnpi(void) { return const_ln_pi; }
jHighPrecisionFloat_mpfr jHighPrecisionFloat_mpfr::ln2(void) { return jHighPrecisionFloat_mpfr(mpfr_const_log2); }
jHighPrecisionFloat_mpfr jHighPrecisionFloat_mpfr::epsilon(void) { return const_epsilon; }
double jHighPrecisionFloat_mpfr::epsilonAsDouble(void) { return const_epsilon_d; }
double jHighPrecisionFloat_mpfr::RoundingError(double newVal, double inErr)
{
	if (inErr != inErr)
	{
		printf("Asked to apply error of %lf for doubleVal %le\n", inErr, newVal);
		ALWAYS_ASSERT(0);
	}
	return sqrt(SQUARE(inErr) + SQUARE(newVal * epsilonAsDouble()));
}

jHighPrecisionFloat_mpfr gsl_sf_lnpoch(const jHighPrecisionFloat_mpfr &a, const jHighPrecisionFloat_mpfr &x)
{
	// I have not yet considered whether this has awkward points where the two terms are similar in value and we lose accuracy,
	// but for now I will just implement this in the naive manner.
	// The Pochhammer symbol is defined as (a)_x := Gamma[a + x]/Gamma[a]
	return ln_gamma_function(a+x) - ln_gamma_function(a);
}

jHighPrecisionFloat_mpfr gsl_sf_log_1plusx(const jHighPrecisionFloat_mpfr &x)
{
	/*	Error calculation: d/dx(log(1+x)) = d/du(log(u)) = 1/u = 1/(1+x)	*/
	return jHighPrecisionFloat_mpfr(mpfr_log1p, x, x.doubleErr() / fabs(1 + x.doubleVal()), false);
}

int gsl_sf_legendre_sphPlm_array(const int lmax, int m, const jreal &x, jreal *result_array)
{
	// GSL source code converted to work with jreal types
	ALWAYS_ASSERT((m>=0) && (lmax >= m) && (x >= jreal(-1)) && (x <= jreal(1)));

	if (m > 0 && (x == 1 || x == -1))
	{
		for (int ell = m; ell <= lmax; ell++)
			result_array[ell-m] = 0;
		return GSL_SUCCESS;
	}
	else
	{
		jreal y_mm;
		jreal y_mmp1;
		
		if (m == 0)
		{
			y_mm   = 1/(2*sqrt(jreal::pi()));          /* Y00 = 1/sqrt(4pi) */
			y_mmp1 = x * sqrt(jreal(3)) * y_mm;
		}
		else
		{
			/* |x| < 1 here */
			jreal lnpre;
			const jreal sgn = ( GSL_IS_ODD(m) ? -1 : 1);
			jreal lncirc = gsl_sf_log_1plusx(-x*x);
			jreal lnpoch = gsl_sf_lnpoch(m, jreal(0.5));  /* Gamma(m+1/2)/Gamma(m) */
			lnpre = -const_ln_pi / 4 + (lnpoch + m*lncirc) / 2;
			y_mm   = sqrt((2+frac<jreal>(1, m)) / (4*jreal::pi())) * sgn * exp(lnpre);
			y_mmp1 = x * sqrt(jreal(2*m + 3)) * y_mm;
		}
		
		if (lmax == m)
		{
			result_array[0] = y_mm;
			return GSL_SUCCESS;
		}
		else if (lmax == m + 1)
		{
			result_array[0] = y_mm;
			result_array[1] = y_mmp1;
			return GSL_SUCCESS;
		}
		else
		{
			jreal y_ell;
			int ell;
			
			result_array[0] = y_mm;
			result_array[1] = y_mmp1;
			
			/* Compute Y_l^m, l > m+1, upward recursion on l. */
			for (ell=m+2; ell <= lmax; ell++)
			{
				const jreal rat1 = frac<jreal>(ell-m, ell+m);
				const jreal rat2 = frac<jreal>(ell-m-1, ell+m-1);
				const jreal factor1 = sqrt(rat1*(2*ell+1)*(2*ell-1));
				const jreal factor2 = sqrt(rat1*rat2*frac<jreal>(2*ell+1, 2*ell-3));
				y_ell = (x*y_mmp1*factor1 - (ell+m-1)*y_mm*factor2) / (ell-m);
				y_mm   = y_mmp1;
				y_mmp1 = y_ell;
				result_array[ell-m] = y_ell;
			}
		}
		return GSL_SUCCESS;
	}
}

int gsl_sf_bessel_Jn_array(int nmin, int nmax, jreal &x, jreal *result_array)
{
	// TODO: I need to work out how this handles underflow and whether this is a problem.
	// The problem will be that mpfr_jn will return zero (or NaN, or something), I presume...
	ALWAYS_ASSERT(nmin >= 0 && nmax >= nmin);

	if (x == 0)
	{
		for (int n = nmax; n >= nmin; n--)
			result_array[n-nmin] = 0;
		if (nmin == 0)
			result_array[0] = 1;
	}
	else
	{
		jreal Jnp1 = besselJn(nmax+1, x);
		jreal Jn = besselJn(nmax, x);
		jreal Jnm1;

		for (int n = nmax; n >= nmin; n--)
		{
			result_array[n-nmin] = Jn;
			Jnm1 = -Jnp1 + 2*n/x * Jn;
			Jnp1 = Jn;
			Jn   = Jnm1;
		}
	}
	return GSL_SUCCESS;
}

#if 0
// May be needed?
jreal gsl_sf_bessel_j0(const jreal &x)
{
	// For now I do not do any special-case handling of very small x - I just let the error accumulate
	if (x == 0)
		return jreal(1);
	return sin(x) / x;
}


jreal gsl_sf_bessel_j1(const jreal &x)
{
	// For now I do not do any special-case handling of very small x - I just let the error accumulate
	if (x == 0)
		return 0;
	return (sin(x) / x - cos(x)) / x;
}


jreal gsl_sf_bessel_j2(const jreal &x)
{
	// For now I do not do any special-case handling of very small x - I just let the error accumulate
	if (x == 0)
		return 0;
	const jreal f = (3/(x*x) - 1);
	return (f * sin(x) - 3*cos(x)/x)/x;
}

jreal gsl_sf_bessel_jl(const int lmax, const jreal &x)
{
	// Not yet implemented
}
#endif

//int gsl_sf_bessel_jl_array(const int lmax, const jreal x, jreal *result_array)
template<class xType> void t_sf_bessel_jl_array(const int lmax, const xType &x, xType *result_array)
{
	// I implement Steed's method for this (taken from GSL) simply because it seems to require less code!
	// Will have to see how much time it takes to execute...
	// I haven't yet generalized this to complex x. A google suggests this should be possible though...
	ALWAYS_ASSERT(lmax >= 0);

	if (x == xType(0))
	{
		result_array[0] = xType(1);
		for (int j = 1; j <= lmax; j++)
			result_array[j] = xType(0);
	}
	// Not doing Taylor series - so in principle we could get error values exploding for exceptionally tiny x
	else
	{
		/* Steed/Barnett algorithm [Comp. Phys. Comm. 21, 297 (1981)] */
		xType x_inv = xType(1)/x;
		xType W = xType(2)*x_inv;
		xType F = xType(1);
		xType FP = xType(lmax+1) * x_inv;
		xType B = xType(2)*FP + x_inv;
		xType end = fabs(B) + xType(20000)*W;
		xType D = xType(1)/B;
		xType del = -D;
		
		FP += del;
		
		/* continued fraction */
		do
		{
			B += W;
			D = xType(1)/(B-D);
			del *= (B*D - 1);
			FP += del;
			if (D < 0)
				F = -F;
			ALWAYS_ASSERT(fabs(B) <= end);		// Test if maximum iterations exceeded
		}
		while (fabs(del) >= fabs(FP) * jHighPrecisionFloat_mpfr::epsilon());
		
		// The value should hopefully be accurate to the precision we are working to?
		// TODO: should try and confirm that...
		F.setErr(F.doubleVal() * jHighPrecisionFloat_mpfr::epsilonAsDouble());
		
		FP *= F;
		
		if (lmax > 0)
		{
			/* downward recursion */
			jreal XP2 = FP;
			jreal PL = lmax * x_inv;
			int L  = lmax;
			result_array[lmax] = F;
			for (int LP = 1; LP<=lmax; LP++)
			{
				result_array[L-1] = PL * result_array[L] + XP2;
				FP = PL*result_array[L-1] - result_array[L];
				XP2 = FP;
				PL -= x_inv;
				--L;
			}
			F = result_array[0];
		}
		
		/* normalization */
		W = x_inv / hypot(FP, F);
		result_array[0] = W*F;
		if (lmax > 0)
		{
			for (int L=1; L<=lmax; L++)
				result_array[L] *= W;
		}
		
	}
}

int gsl_sf_bessel_jl_array(const int lmax, const jreal &x, jreal *result_array)
{
	t_sf_bessel_jl_array<jreal>(lmax, x, result_array);
	return GSL_SUCCESS;
}

jComplexVectorR z_bessel_jl_array(const int lmax, const jComplexR &z)
{
	// TODO: I haven't yet generalized the Steed/Barnett code to complex x. A google suggests this should be possible though...
	jComplexVectorR result(lmax + 1);
	if (imag(z) == 0)
	{
		std::vector<jreal> resultReal(lmax + 1);
		t_sf_bessel_jl_array<jreal>(lmax, real(z), &resultReal[0]);
		for (int i = 0; i < lmax + 1; i++)
			result[i] = resultReal[i];
		return result;
	}
	else
	{
		// I have not properly implemented this for complex z.
		// I don't expect to encounter this during high-precision calculations, but we should
		// provide it as a function
		// For now, just call through to the low-precision version, but make it clear that we have a large uncertainty.
		std::vector<jComplex> result_lp = z_bessel_jl_array(lmax, AllowPrecisionLossReadingValue(z));
		for (int i = 0; i < lmax + 1; i++)
			result[i] = jComplexR(jreal(result_lp[i].real()), jreal(result_lp[i].imag()));
		return result;
	}
}

jComplexR z_bessel_jl(const int l, const jComplexR &x)
{
	if (imag(x) == 0)
	{
		// This is pretty inefficient for now - doing the full array generation just to get j_l.
		// It's not as bad as it might seem though - it's generating that first value that takes most of the time,
		// and then the others are easily obtained by downward recurrence.
		std::vector<jreal> resultReal(l + 1);
		t_sf_bessel_jl_array<jreal>(l, real(x), &resultReal[0]);
		return resultReal[l];
	}
	else
	{
		// I have not properly implemented this for complex x.
		// I don't expect to encounter this during high-precision calculations, but we should
		// provide it as a function
		// For now, just call through to the low-precision version, but make it clear that we have a large uncertainty.
		jComplex result_lp = z_bessel_jl(l, AllowPrecisionLossReadingValue(x));
		return jComplexR(jreal(result_lp.real()), jreal(result_lp.imag()));
	}
}
