/*
 *	jComplexPairAsVector256.h
 *
 *  Class representing two complex numbers, storing the values as a single 256-bit AVX vector.
 *	This currently supports most common arithmetic operations, as well as a few special functions
 *	(e.g. GetMulWithConjY) to perform special operations which can have particularly efficient implementations.
 */

#ifndef __JCOMPLEX_PAIR_AS_VECTOR_256_H__
#define __JCOMPLEX_PAIR_AS_VECTOR_256_H__

#include "jComplex.h"
#include "VectorTypes.h"
#include "VectorFunctions.h"

class jComplexPairAsVector256
{
	// This class represents two complex numbers, storing them in a single 256-bit AVX vector
  protected:
	__m256d __ab;
	
  public:

	jComplexPairAsVector256() { }

	__m256d ab(void) const { return __ab; }
	jComplex a(void) const { return jComplex(__ab[0], __ab[1]); }
	jComplex b(void) const { return jComplex(__ab[2], __ab[3]); }
//	vDouble re(void) const { return _mm256_extractf128_pd(__ab, 0); }		// Needs correcting if present
//	vDouble im(void) const { return _mm256_extractf128_pd(__ab, 1); }

	explicit jComplexPairAsVector256(const jComplex &inZ) { _mm256_set_pd(real(inZ), real(inZ), imag(inZ), imag(inZ)); }
	jComplexPairAsVector256(const jComplexPairAsVector256 &inAB) { __ab = inAB.ab(); }
	jComplexPairAsVector256(jComplex inA, jComplex inB) { _mm256_set_pd(real(inA), real(inB), imag(inA), imag(inB)); }
	jComplexPairAsVector256(__m256d inAB) { __ab = inAB; }
	
	void SetA(jComplex inA) { _mm256_set_pd(real(inA), __ab[1], imag(inA), __ab[3]); }
	void SetB(jComplex inB) { _mm256_set_pd(__ab[0], real(inB), __ab[2], imag(inB)); }
	
	jComplexPairAsVector256 conj(void) const { return jComplexPairAsVector256(_mm256_xor_pd(__ab, (__m256d) { 0.0, 0.0, -0.0, -0.0 })); }

	jComplexPairAsVector256& operator += (const jComplexPairAsVector256 &n) { __ab = _mm256_add_pd(__ab, n.ab()); return *this; }
	jComplexPairAsVector256 operator + (const jComplexPairAsVector256 &n) const { return jComplexPairAsVector256(*this) += n; }
	jComplexPairAsVector256& operator -= (const jComplexPairAsVector256 &n) { __ab = _mm256_sub_pd(__ab, n.ab()); return *this; }
	jComplexPairAsVector256 operator - (const jComplexPairAsVector256 &n) const { return jComplexPairAsVector256(*this) -= n; }
	jComplexPairAsVector256& operator *= (const jComplexPairAsVector256 &n)
	{
		****
		vDouble rere = _mm_mul_pd(__re, n.re());
		vDouble reim = _mm_mul_pd(__re, n.im());
		vDouble imre = _mm_mul_pd(__im, n.re());
		vDouble imim = _mm_mul_pd(__im, n.im());
		__re = _mm_sub_pd(rere, imim);
		__im = _mm_add_pd(reim, imre);
		return *this;
	}
	jComplexPairAsVector256 operator * (const jComplexPairAsVector256 &n) const { return jComplexPairAsVector256(*this) *= n; }
	jComplexPairAsVector256 GetMulWithConjY(const jComplexPairAsVector256 &y) const//TEST
	{
		// Return the pairwise product of 'this' with the complex conjugate of y
		****
		__m256d rere_imim = _mm256_mul_pd(__ab, y.ab());				// { x1.real() * y1.real(), x2.real() * y2.real(), x1.imag() * y1.imag(), x2.imag() * y2.imag() }
		__m256d reim_imre = _mm256_mul_pd(__ab, y.GetSwappedReIm());	// { x1.real() * y1.imag(), x2.real() * y2.imag(), x1.imag() * y1.real(), x2.imag() * y2.real() }
		
		vDouble rere = _mm_mul_pd(__re, y.re());
		vDouble reim = _mm_mul_pd(__re, y.im());
		vDouble imre = _mm_mul_pd(__im, y.re());
		vDouble imim = _mm_mul_pd(__im, y.im());
		return jComplexPairAsVector256(_mm_add_pd(rere, imim), _mm_sub_pd(imre, reim));
	}
	jComplexPairAsVector256& operator /= (const jComplexPairAsVector256 &n)
	{
		****
		jComplexPairAsVector256 conjMul = GetMulWithConjY(n);
		vDouble nNorm = _mm_add_pd(_mm_mul_pd(n.re(), n.re()), _mm_mul_pd(n.im(), n.im()));
		__re = _mm_div_pd(conjMul.re(), nNorm);
		__im = _mm_div_pd(conjMul.im(), nNorm);
		return *this;
	}
	jComplexPairAsVector256 operator / (const jComplexPairAsVector256 &n) const { return jComplexPairAsVector256(*this) /= n; }

	jComplexPairAsVector256& operator += (const jComplex &n) { operator+=(jComplexPairAsVector256(n)); return *this; }
	jComplexPairAsVector256 operator + (const jComplex &n) const { return jComplexPairAsVector256(*this) += n; }
	jComplexPairAsVector256& operator -= (const jComplex &n) { operator-=(jComplexPairAsVector256(n)); return *this; }
	jComplexPairAsVector256 operator - (const jComplex &n) const { return jComplexPairAsVector256(*this) -= n; }
	jComplexPairAsVector256& operator *= (const jComplex &n) { operator*=(jComplexPairAsVector256(n)); return *this; }
	jComplexPairAsVector256 operator * (const jComplex &n) const { return jComplexPairAsVector256(*this) *= n; }
	jComplexPairAsVector256& operator /= (const jComplex &n) { operator/=(jComplexPairAsVector256(n)); return *this; }
	jComplexPairAsVector256 operator / (const jComplex &n) const { return jComplexPairAsVector256(*this) /= n; }

	jComplexPairAsVector256& operator += (const double &n) { __re = _mm256_add_pd(__ab, (__mm256d){n, n, 0, 0}); return *this; }
	jComplexPairAsVector256 operator + (const double &n) const { return jComplexPairAsVector256(*this) += n; }
	jComplexPairAsVector256& operator -= (const double &n) { __re = _mm256_sub_pd(__ab, (__mm256d){n, n, 0, 0}); return *this; }
	jComplexPairAsVector256 operator - (const double &n) const { return jComplexPairAsVector256(*this) -= n; }
	jComplexPairAsVector256& operator *= (double n) { __re = _mm256_mul_pd(__ab, (__mm256d){n, n, n, n}); return *this; }
	jComplexPairAsVector256 operator * (double n) const { return jComplexPairAsVector256(*this) *= n; }
	jComplexPairAsVector256& operator /= (double n) { __re = _mm256_div_pd(__ab, (__mm256d){n, n, n, n}); return *this; }
	jComplexPairAsVector256 operator / (double n) const { return jComplexPairAsVector256(*this) /= n; }

	double SumReal(void) const
	{
		// Return the sum of the real parts of the two numbers
		__m256d temp = _mm256_hadd_pd(__ab, __ab);		// VHADD x, p, q => x = {p[0]+p[1], q[0]+q[1], p[2]+p[3], q[2]+q[3]}
		return temp[0];
	}
	double SumImag(void) const
	{
		// Return the sum of the imaginary parts of the two numbers
		__m256d temp = _mm256_hadd_pd(__ab, __ab);		// VHADD x, p, q => x = {p[0]+p[1], q[0]+q[1], p[2]+p[3], q[2]+q[3]}
		return temp[2];
	}
	double SumNorm(void) const //TEST
	{
		// Return the sum of the norms of the two numbers
		// NOTE: as per complex::norm, we use norm in the sense of the *squared* magnitude,
		// which is not the standard mathematical definition, but is what is used in complex::norm.
		__m256d sq = _mm256_mul_pd(__ab, __ab);
		__m256d temp = _mm256_hadd_pd(sq, sq);		// VHADD x, p, q => x = {p[0]+p[1], q[0]+q[1], p[2]+p[3], q[2]+q[3]}
		__m128d sum_high = _mm256_extractf128_pd(sum, 1);
		// add upper 128 bits of sum to its lower 128 bits
		__m128d result = _mm_add_pd(sum_high, _mm256_castpd256_pd128(sum));
		// lower 64 bits of result contain the sum of x1[0], x1[1], x1[2], x1[3]
		// upper 64 bits of result contain the same sum (would be sum of second parameter
		// if we gave two separate parameters to the HADD)
		return result[0];
	}
	jComplex SumAcross(void) const	//TEST
	{
		// Return the sum of the two values as a single complex number
		__m256d temp = _mm256_hadd_pd(__ab, __ab);		// VHADD x, p, q => x = {p[0]+p[1], q[0]+q[1], p[2]+p[3], q[2]+q[3]}
		return jComplex(temp[0], temp[2]);
	}
	jComplexPairAsVector256 GetNegative(void) const { return jComplexPairAsVector256(_mm256_xor_pd(__ab, (__m256d) { -0.0, -0.0, -0.0, -0.0 })); }		// Return { -a, -b }
	jComplexPairAsVector256 GetNegativeOfSecondOnly(void) const { return jComplexPairAsVector256(_mm256_xor_pd(__ab, (__m256d) { 0.0, -0.0, 0.0, -0.0 })); }		// Return { a, -b }
	jComplexPairAsVector256 GetSwappedPairs(void) const { return jComplexPairAsVector256(__builtin_shufflevector(__ab, __ab, 1, 0, 3, 2); }	// Return { b, a } given { a, b }		//TEST
	jComplexPairAsVector256 GetSwappedReIm(void) const { return jComplexPairAsVector256(_mm256_permute_pd(__ab, (2 + 3*4 + 0*16 + 1*64))); }	// Return { im(a), im(b), re(a), re(b) } given { a, b }		//TEST!!

	void Print(void) const
	{
		printf("{");
		::Print(a());
		printf(", ");
		::Print(b());
		printf("}");
	}
};

inline jComplexPairAsVector256 operator*(const double l, const jComplexPairAsVector256 &r)
{
	return r * l;
}

inline jComplexPairAsVector256 operator*(const jComplex l, const jComplexPairAsVector256 &r)
{
	return r * l;
}

inline jComplexPairAsVector256 operator/(const double l, const jComplexPairAsVector256 &r)
{
	return jComplexPairAsVector256(l) / r;
}

inline jComplexPairAsVector256 operator/(const jComplex l, const jComplexPairAsVector256 &r)
{
	return jComplexPairAsVector256(l) / r;
}

inline jComplexPairAsVector256 operator-(const double l, const jComplexPairAsVector256 &r)
{
	return jComplexPairAsVector256(l) - r;
}

inline jComplexPairAsVector256 operator-(const jComplexPairAsVector256 &r)
{
	return r.GetNegative();
}

inline jComplexPairAsVector256 operator+(const jComplexPairAsVector256 &r)
{
	return r;
}

inline jComplexPairAsVector256 operator+(const double l, const jComplexPairAsVector256 &r)
{
	return r + l;
}

inline jComplexPairAsVector256 conj(const jComplexPairAsVector256 &z)
{
	return z.conj();
}

inline jComplexPairAsVector256 re_part(const jComplexPairAsVector256 &x) { return jComplexPairAsVector256(_mm256_and_pd(__ab, (__v8si) { -1, -1, -1, -1, 0, 0, 0, 0 })); }//TEST
inline jComplexPairAsVector256 im_part(const jComplexPairAsVector256 &x) { return jComplexPairAsVector256(_mm256_and_pd(__ab, (__v8si) { 0, 0, 0, 0, -1, -1, -1, -1 })); }//TEST
inline double SumReal(const jComplexPairAsVector256 &x) { return x.SumReal(); }
inline double SumNorm(const jComplexPairAsVector256 &x) { return x.SumNorm(); }
inline jComplex SumAcross(const jComplexPairAsVector256 &x) { return x.SumAcross(); }
inline jComplexPairAsVector256 MulXConjY(const jComplexPairAsVector256 &x, const jComplexPairAsVector256 &y) { return x.GetMulWithConjY(y); }

inline jComplexPairAsVector256 SwapPairs(const jComplexPairAsVector256 &x) { return jComplexPairAsVector256(x.GetSwappedPairs()); }

#endif
