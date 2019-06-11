/*
 *	jComplexPairAsVector.h
 *
 *  Copyright 2011-2015 Jonathan Taylor. All rights reserved.
 *
 *  Class representing two complex numbers, storing the values as two altivec/SSE vectors,
 *	one for the real parts and one for the complex parts. This currently supports most common
 *	arithmetic operations, as well as a few special functions (e.g. GetMulWithConjY) to perform
 *	special operations which can have particularly efficient implementations.
 */

#ifndef __JCOMPLEX_PAIR_AS_VECTOR_H__
#define __JCOMPLEX_PAIR_AS_VECTOR_H__

#include "jComplex.h"
#include "VectorTypes.h"
#include "VectorFunctions.h"

class jComplexPairAsVector
{
	// This class represents two complex numbers
	// This implementation stores the complex numbers in split form
  protected:
	vDouble __re, __im;
	
  public:

	jComplexPairAsVector() { }
	static jComplexPairAsVector zero(void) { return jComplexPairAsVector(0, 0, 0, 0); }

	jComplex a(void) const { return jComplex(vLower(__re), vLower(__im)); }
	jComplex b(void) const { return jComplex(vUpper(__re), vUpper(__im)); }
	vDouble re(void) const { return __re; }
	vDouble im(void) const { return __im; }
    double a_r(void) const { return __re[0]; }
    double a_i(void) const { return __im[0]; }
    double b_r(void) const { return __re[1]; }
    double b_i(void) const { return __im[1]; }

	explicit jComplexPairAsVector(const jComplex &inZ) { __re = (vDouble){real(inZ), real(inZ)}; __im = (vDouble){imag(inZ), imag(inZ)}; }
	jComplexPairAsVector(const jComplexPairAsVector &inAB) { __re = inAB.re(); __im = inAB.im(); }
	jComplexPairAsVector(jComplex inA, jComplex inB) { __re = (vDouble){real(inA), real(inB)}; __im = (vDouble){imag(inA), imag(inB)}; }
    jComplexPairAsVector(double ar, double ai, double br, double bi) { __re = (vDouble){ar, br}; __im = (vDouble){ai, bi}; }
	jComplexPairAsVector(vDouble inRe, vDouble inIm) { __re = inRe; __im = inIm; }
	
	void SetA(jComplex inA) { __re = (vDouble){real(inA), real(b())}; __im = (vDouble){imag(inA), imag(b())}; }
	void SetB(jComplex inB) { __re = (vDouble){real(a()), real(inB)}; __im = (vDouble){imag(a()), imag(inB)}; }
	
	jComplexPairAsVector conj(void) const { return jComplexPairAsVector(__re, vNegate(__im)); }

	jComplexPairAsVector& operator += (const jComplexPairAsVector &n) { __re = _mm_add_pd(__re, n.re()); __im = _mm_add_pd(__im, n.im()); return *this; }
	jComplexPairAsVector operator + (const jComplexPairAsVector &n) const { return jComplexPairAsVector(*this) += n; }
	jComplexPairAsVector& operator -= (const jComplexPairAsVector &n) { __re = _mm_sub_pd(__re, n.re()); __im = _mm_sub_pd(__im, n.im()); return *this; }
	jComplexPairAsVector operator - (const jComplexPairAsVector &n) const { return jComplexPairAsVector(*this) -= n; }
	jComplexPairAsVector& operator *= (const jComplexPairAsVector &n)
	{
		vDouble rere = _mm_mul_pd(__re, n.re());
		vDouble reim = _mm_mul_pd(__re, n.im());
		vDouble imre = _mm_mul_pd(__im, n.re());
		vDouble imim = _mm_mul_pd(__im, n.im());
		__re = _mm_sub_pd(rere, imim);
		__im = _mm_add_pd(reim, imre);
		return *this;
	}
	jComplexPairAsVector operator * (const jComplexPairAsVector &n) const { return jComplexPairAsVector(*this) *= n; }
	jComplexPairAsVector GetMulWithConjY(const jComplexPairAsVector &y) const
	{
		// Return the pairwise product of 'this' with the complex conjugate of y
		vDouble rere = _mm_mul_pd(__re, y.re());
		vDouble reim = _mm_mul_pd(__re, y.im());
		vDouble imre = _mm_mul_pd(__im, y.re());
		vDouble imim = _mm_mul_pd(__im, y.im());
		return jComplexPairAsVector(_mm_add_pd(rere, imim), _mm_sub_pd(imre, reim));
	}
	jComplexPairAsVector& operator /= (const jComplexPairAsVector &n)
	{
		jComplexPairAsVector conjMul = GetMulWithConjY(n);
		vDouble nNorm = _mm_add_pd(_mm_mul_pd(n.re(), n.re()), _mm_mul_pd(n.im(), n.im()));
		__re = _mm_div_pd(conjMul.re(), nNorm);
		__im = _mm_div_pd(conjMul.im(), nNorm);
		return *this;
	}
	jComplexPairAsVector operator / (const jComplexPairAsVector &n) const { return jComplexPairAsVector(*this) /= n; }

	jComplexPairAsVector& operator += (const jComplex &n) { operator+=(jComplexPairAsVector(n)); return *this; }
	jComplexPairAsVector operator + (const jComplex &n) const { return jComplexPairAsVector(*this) += n; }
	jComplexPairAsVector& operator -= (const jComplex &n) { operator-=(jComplexPairAsVector(n)); return *this; }
	jComplexPairAsVector operator - (const jComplex &n) const { return jComplexPairAsVector(*this) -= n; }
	jComplexPairAsVector& operator *= (const jComplex &n) { operator*=(jComplexPairAsVector(n)); return *this; }
	jComplexPairAsVector operator * (const jComplex &n) const { return jComplexPairAsVector(*this) *= n; }
	jComplexPairAsVector& operator /= (const jComplex &n) { operator/=(jComplexPairAsVector(n)); return *this; }
	jComplexPairAsVector operator / (const jComplex &n) const { return jComplexPairAsVector(*this) /= n; }

	jComplexPairAsVector& operator += (const double &n) { __re = _mm_add_pd(__re, (vDouble){n, n}); return *this; }
	jComplexPairAsVector operator + (const double &n) const { return jComplexPairAsVector(*this) += n; }
	jComplexPairAsVector& operator -= (const double &n) { __re = _mm_sub_pd(__re, (vDouble){n, n}); return *this; }
	jComplexPairAsVector operator - (const double &n) const { return jComplexPairAsVector(*this) -= n; }
	jComplexPairAsVector& operator *= (double n) { __re = _mm_mul_pd(__re, (vDouble){n, n}); __im = _mm_mul_pd(__im, (vDouble){n, n}); return *this; }
	jComplexPairAsVector operator * (double n) const { return jComplexPairAsVector(*this) *= n; }
	jComplexPairAsVector& operator /= (double n) { __re = _mm_div_pd(__re, (vDouble){n, n}); __im = _mm_div_pd(__im, (vDouble){n, n}); return *this; }
	jComplexPairAsVector operator / (double n) const { return jComplexPairAsVector(*this) /= n; }

	double SumReal(void) const { return vHAddLower(__re); }		// Return the sum of the real parts of the two numbers
	double SumImag(void) const { return vHAddLower(__im); }		// Return the sum of the imaginary parts of the two numbers
	double SumNorm(void) const
	{
		// Return the sum of the norms of the two numbers
		// NOTE: as per complex::norm, we use norm in the sense of the *squared* magnitude,
		// which is not the standard mathematical definition, but is what is used in complex::norm.
		vDouble re2 = _mm_mul_pd(__re, __re);
		vDouble im2 = _mm_mul_pd(__im, __im);
		vDouble sum2 = _mm_add_pd(re2, im2);
		return vHAddLower(sum2);
	}
	jComplex SumAcross(void) const { return jComplex(SumReal(), SumImag()); }		// Return the sum of the two values as a single complex number
    jComplexPairAsVector Negated(double ar, double ai, double br, double bi) const { return jComplexPairAsVector(_mm_mul_pd(__re, (vDouble){ar, br}), _mm_mul_pd(__im, (vDouble){ai, bi})); }
    jComplexPairAsVector NegatedUsingXORWith(jComplexPairAsVector neg) const { return jComplexPairAsVector(_mm_xor_pd(__re, neg.re()), _mm_xor_pd(__im, neg.im())); }   // Expects a vector containing either 0.0 or -0.0 entries. Uses the XOR operator to negate self according to that pattern
	jComplexPairAsVector GetNegative(void) const { return jComplexPairAsVector(vNegate(__re), vNegate(__im)); }		// Return the negated values of a and b
	jComplexPairAsVector GetNegativeOfSecondOnly(void) const { return jComplexPairAsVector(_mm_xor_pd(__re, (vDouble) { 0.0, -0.0 }), _mm_xor_pd(__im, (vDouble) { 0.0, -0.0 })); }		// Return the negated value of b, leaving a unchanged
    jComplexPairAsVector GetNegativeOfFirstOnly(void) const { return jComplexPairAsVector(_mm_xor_pd(__re, (vDouble) { -0.0, 0.0 }), _mm_xor_pd(__im, (vDouble) { -0.0, 0.0 })); }		// Return the negated value of a, leaving b unchanged
	jComplexPairAsVector GetSwappedPairs(void) const { return jComplexPairAsVector(vSwapD(__re), vSwapD(__im)); }	// Return { b, a } given { a, b }

	void Print(const char *suffix = "") const
	{
		printf("{");
		::Print(a());
		printf(", ");
		::Print(b());
		printf("}%s", suffix);
	}
};

inline jComplexPairAsVector operator*(const double l, const jComplexPairAsVector &r)
{
	return r * l;
}

inline jComplexPairAsVector operator*(const jComplex l, const jComplexPairAsVector &r)
{
	return r * l;
}

inline jComplexPairAsVector operator/(const double l, const jComplexPairAsVector &r)
{
	return jComplexPairAsVector(l / r.a(), l / r.b());
}

inline jComplexPairAsVector operator/(const jComplex l, const jComplexPairAsVector &r)
{
	return jComplexPairAsVector(l / r.a(), l / r.b());
}

inline jComplexPairAsVector operator-(const double l, const jComplexPairAsVector &r)
{
	return jComplexPairAsVector(l - r.a(), l - r.b());
}

inline jComplexPairAsVector operator-(const jComplexPairAsVector &r)
{
	return r.GetNegative();
}

inline jComplexPairAsVector operator+(const jComplexPairAsVector &r)
{
	return r;
}

inline jComplexPairAsVector operator+(const double l, const jComplexPairAsVector &r)
{
	return r + l;
}

inline jComplexPairAsVector conj(const jComplexPairAsVector &z)
{
	return z.conj();
}

inline jComplexPairAsVector re_part(const jComplexPairAsVector &x) { return jComplexPairAsVector(x.re(), vZeroD); }
inline jComplexPairAsVector im_part(const jComplexPairAsVector &x) { return jComplexPairAsVector(x.im(), vZeroD); }
inline double SumReal(const jComplexPairAsVector &x) { return x.SumReal(); }
inline double SumNorm(const jComplexPairAsVector &x) { return x.SumNorm(); }
inline jComplex SumAcross(const jComplexPairAsVector &x) { return x.SumAcross(); }
inline jComplexPairAsVector MulXConjY(const jComplexPairAsVector &x, const jComplexPairAsVector &y) { return x.GetMulWithConjY(y); }

inline jComplexPairAsVector SwapPairs(const jComplexPairAsVector &x) { return jComplexPairAsVector(x.GetSwappedPairs()); }

#endif
