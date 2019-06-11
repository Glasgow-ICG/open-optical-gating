/*
 *	jHighPrecisionFloat_mpfr.h
 *
 *  Copyright 2011-2015 Jonathan Taylor. All rights reserved.
 *
 *  Class representing a floating-point number, but potentially to higher (or lower) precision
 *	than supported by the ubiquitous 'double' type.
 *
 *	This implementation relies on the MPFR library to implement the underlying high-precision arithmetic.
 */

#ifndef __JHIGH_PRECISION_FLOAT_MPFR_H__
#define __JHIGH_PRECISION_FLOAT_MPFR_H__

#define HIGH_PRECISION_REAL 1

#include <src/mpfr.h>

// TODO: all these need redefining once I have a proper high precision implementation
#define J_DBL_EPSILON jreal(GSL_DBL_EPSILON)
#define J_SQRT_DBL_EPSILON jreal(GSL_SQRT_DBL_EPSILON)
#define J_ROOT4_DBL_EPSILON jreal(GSL_ROOT4_DBL_EPSILON)
#define J_ROOT6_DBL_EPSILON jreal(GSL_ROOT6_DBL_EPSILON)
#define J_DBL_MIN jreal(GSL_DBL_MIN)
#define J_DBL_MAX jreal(GSL_DBL_MAX)
#define J_LOG_DBL_MIN jreal(GSL_LOG_DBL_MIN)
#define J_LOG_DBL_MAX jreal(GSL_LOG_DBL_MAX)
#define J_SQRT_DBL_MIN jreal(GSL_SQRT_DBL_MIN)
#define J_SQRT_DBL_MAX jreal(GSL_SQRT_DBL_MAX)
#define J_POSINF jreal::posinf()
#define J_NAN jreal::nan()
#define J_LNPI jreal::lnpi()
#define J_LN2 jreal::ln2()
#define J_REAL_EPSILON jreal::epsilon()

class jHighPrecisionFloat_mpfr
{
	// This class represents a high-precision floating-point number, with associated error on accuracy
public:
	mpfr_t		__val;
	double		__err;

public:
	static const mpfr_rnd_t rounding;
	static const mpfr_prec_t precision;
	
	jHighPrecisionFloat_mpfr();
	virtual ~jHighPrecisionFloat_mpfr();
	jHighPrecisionFloat_mpfr &operator =(const jHighPrecisionFloat_mpfr &copy);
	
	double doubleVal(void) const { return mpfr_get_d(__val, rounding); }
	double doubleErr(void) const { return __err; }
	void setErr(double inErr) { __err = inErr; }
	
	explicit jHighPrecisionFloat_mpfr(double inVal);
	explicit jHighPrecisionFloat_mpfr(double inVal, double inErr);

	// For no-argument functions, e.g. constants like pi
	explicit jHighPrecisionFloat_mpfr(int (*func)(mpfr_ptr, mpfr_rnd_t));
	// For single-argument functions e.g. sin
	explicit jHighPrecisionFloat_mpfr(int (*func)(mpfr_ptr, mpfr_srcptr, mpfr_rnd_t), const jHighPrecisionFloat_mpfr &srcVal, double inErr, bool noRoundingWillOccur);
	// For two-argument functions e.g. atan2
	explicit jHighPrecisionFloat_mpfr(int (*func)(mpfr_ptr, mpfr_srcptr, mpfr_srcptr, mpfr_rnd_t), const jHighPrecisionFloat_mpfr &x, const jHighPrecisionFloat_mpfr &y, double inErr, bool noRoundingWillOccur);
	// Sometimes necessary to do things semi-manually like this
	explicit jHighPrecisionFloat_mpfr(mpfr_ptr inVal, double inErr);
	
	jHighPrecisionFloat_mpfr(const jHighPrecisionFloat_mpfr &obj);
	jHighPrecisionFloat_mpfr(int inVal);
	jHighPrecisionFloat_mpfr(long inVal);		// Rarely used, but I do occasionally use this
//	jHighPrecisionFloat_mpfr(long long inVal) { __val = inVal; __err = 0.0; }
	jHighPrecisionFloat_mpfr(const char *numString, double inErr);
	
	const mpfr_t &mpfrVal(void) const { return __val; }
	mpfr_t &mpfrVal_nonConst(void) { return __val; }
	
	jHighPrecisionFloat_mpfr& operator += (const jHighPrecisionFloat_mpfr &n);
	jHighPrecisionFloat_mpfr operator + (const jHighPrecisionFloat_mpfr &n) const { return jHighPrecisionFloat_mpfr(*this) += n; }
	jHighPrecisionFloat_mpfr& operator -= (const jHighPrecisionFloat_mpfr &n);
	jHighPrecisionFloat_mpfr operator - (const jHighPrecisionFloat_mpfr &n) const { return jHighPrecisionFloat_mpfr(*this) -= n; }
	jHighPrecisionFloat_mpfr& operator *= (const jHighPrecisionFloat_mpfr &n);
	jHighPrecisionFloat_mpfr operator * (const jHighPrecisionFloat_mpfr &n) const { return jHighPrecisionFloat_mpfr(*this) *= n; }
	jHighPrecisionFloat_mpfr& operator /= (const jHighPrecisionFloat_mpfr &n);
	jHighPrecisionFloat_mpfr operator / (const jHighPrecisionFloat_mpfr &n) const { return jHighPrecisionFloat_mpfr(*this) /= n; }
	
	jHighPrecisionFloat_mpfr &NegateThis(void)
	{
		mpfr_neg(__val, __val, rounding);
		return (*this);
	}
	jHighPrecisionFloat_mpfr GetNegative(void) const { return jHighPrecisionFloat_mpfr(*this).NegateThis(); }

	static jHighPrecisionFloat_mpfr dbl_max(void);
	static jHighPrecisionFloat_mpfr dbl_min(void);
	static jHighPrecisionFloat_mpfr nan(void);
	static jHighPrecisionFloat_mpfr posinf(void);
	static jHighPrecisionFloat_mpfr pi(void);
	static jHighPrecisionFloat_mpfr lnpi(void);
	static jHighPrecisionFloat_mpfr ln2(void);
	static jHighPrecisionFloat_mpfr epsilon(void);
	static double epsilonAsDouble(void);
	static double RoundingError(double newVal, double inErr);

	void Print(const char *suffix = "") const
	{
		mpfr_printf("%.38RNe", __val);
		printf("±%.2le", __err);
		printf("%s", suffix);
	}
};

inline jHighPrecisionFloat_mpfr operator-(const jHighPrecisionFloat_mpfr &r)
{
	return r.GetNegative();
}

inline jHighPrecisionFloat_mpfr operator+(const jHighPrecisionFloat_mpfr &r)
{
	return r;
}

inline jHighPrecisionFloat_mpfr operator*(const int l, const jHighPrecisionFloat_mpfr &r)
{
	return r * l;
}

inline jHighPrecisionFloat_mpfr operator/(const int l, const jHighPrecisionFloat_mpfr &r)
{
	return jHighPrecisionFloat_mpfr(l) / r;
}

inline jHighPrecisionFloat_mpfr operator+(const int l, const jHighPrecisionFloat_mpfr &r)
{
	return jHighPrecisionFloat_mpfr(l) + r;
}

inline jHighPrecisionFloat_mpfr operator-(const int l, const jHighPrecisionFloat_mpfr &r)
{
	return jHighPrecisionFloat_mpfr(l) - r;
}

inline bool operator == (const jHighPrecisionFloat_mpfr &x, const jHighPrecisionFloat_mpfr &y)
{
	return (mpfr_cmp(x.mpfrVal(), y.mpfrVal()) == 0);	/* returns 1 if op1 > op2; 0 if op1 == op2; -1 if op1 < op2 */
}

inline bool operator != (const jHighPrecisionFloat_mpfr &x, const jHighPrecisionFloat_mpfr &y)
{
	return !(x == y);	/* returns 1 if op1 > op2; 0 if op1 == op2; -1 if op1 < op2 */
}

inline bool operator > (const jHighPrecisionFloat_mpfr &x, const jHighPrecisionFloat_mpfr &y)
{
	return (mpfr_cmp(x.mpfrVal(), y.mpfrVal()) == 1);	/* returns 1 if op1 > op2; 0 if op1 == op2; -1 if op1 < op2 */
}

inline bool operator < (const jHighPrecisionFloat_mpfr &x, const jHighPrecisionFloat_mpfr &y)
{
	return (mpfr_cmp(x.mpfrVal(), y.mpfrVal()) == -1);	/* returns 1 if op1 > op2; 0 if op1 == op2; -1 if op1 < op2 */
}

inline bool operator >= (const jHighPrecisionFloat_mpfr &x, const jHighPrecisionFloat_mpfr &y)
{
	return !(x < y);
}

inline bool operator <= (const jHighPrecisionFloat_mpfr &x, const jHighPrecisionFloat_mpfr &y)
{
	return !(x > y);
}

#endif
