/*
 *	jComplexPairAsVector256.h
 *
 *  Copyright 2011-2015 Jonathan Taylor. All rights reserved.
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
#include <immintrin.h>
#include <avxintrin.h>
#include <stdlib.h>

// Horizontal add works as follows:
// VHADD x, p, q => x = {p[0]+p[1], q[0]+q[1], p[2]+p[3], q[2]+q[3]}

class jComplexPairAsVector256
{
	// This class represents two complex numbers, storing them in a single 256-bit AVX vector
  protected:
	__m256d __ab;
	
  public:

	jComplexPairAsVector256() { }
	static jComplexPairAsVector256 zero(void) { return jComplexPairAsVector256(0, 0, 0, 0); }

	void* operator new(size_t size)
	{
		// Override operator new to ensure that the allocated memory is suitably aligned to a 32-byte boundary (as required by some AVX instructions)
		void *storage;
		int result = posix_memalign(&storage, sizeof(jComplexPairAsVector256), size);
		if ((result != 0) || (storage == NULL))
			throw "allocation fail : no free memory";
		return storage;
	}
	
	void* operator new[](size_t size)
	{
		// Override operator new[] to ensure that the allocated memory is suitably aligned to a 32-byte boundary (as required by some AVX instructions)
		void *storage;
		int result = posix_memalign(&storage, sizeof(jComplexPairAsVector256), size);
//		printf("Allocated %p\n", storage);
		if ((result != 0) || (storage == NULL))
			throw "allocation fail : no free memory";
		return storage;
	}

	void operator delete(void *ptr) { free(ptr); }
	void operator delete[](void *ptr) { free(ptr); }

	__m256d ab(void) const { return __ab; }
    const __m256d *abPtr(void) const { return &__ab; }
	jComplex a(void) const { return jComplex(__ab[0], __ab[1]); }
	jComplex b(void) const { return jComplex(__ab[2], __ab[3]); }
	vDouble re(void) const { return _mm256_extractf128_pd(__builtin_shufflevector(__ab, __ab, 0, 2, 1, 3), 0); }
	vDouble im(void) const { return _mm256_extractf128_pd(__builtin_shufflevector(__ab, __ab, 0, 2, 1, 3), 1); }
    double a_r(void) const { return ((double*)&__ab)[0]; }      // Puzzlingly, as of May 2015 this approach gives better code than just __ab[0]
    double a_i(void) const { return ((double*)&__ab)[1]; }
    double b_r(void) const { return ((double*)&__ab)[2]; }
    double b_i(void) const { return ((double*)&__ab)[3]; }

	explicit jComplexPairAsVector256(const double n) { __ab = _mm256_set_pd(n, n, n, n); }		// ***** is this the most efficient way - will the compiler optimize to a broadcast type instruction?
	explicit jComplexPairAsVector256(const jComplex &inZ) { __ab = _mm256_set_pd(imag(inZ), real(inZ), imag(inZ), real(inZ)); }
	jComplexPairAsVector256(const jComplexPairAsVector256 &inAB) { __ab = inAB.ab(); }
	jComplexPairAsVector256(jComplex inA, jComplex inB) { __ab = _mm256_set_pd(imag(inB), real(inB), imag(inA), real(inA)); }
    jComplexPairAsVector256(double ar, double ai, double br, double bi) { __ab = _mm256_set_pd(bi, br, ai, ar); }
	jComplexPairAsVector256(__m256d inAB) { __ab = inAB; }
	jComplexPairAsVector256(__m128d inRe, __m128d inIm) { __ab = _mm256_set_pd(inIm[1], inRe[1], inIm[0], inRe[0]); }		// TODO: temp to be removed (ambiguous)
	
	void SetA(jComplex inA) { __ab = _mm256_set_pd(__ab[3], __ab[2], imag(inA), real(inA)); }
	void SetB(jComplex inB) { __ab = _mm256_set_pd(imag(inB), real(inB), __ab[1], __ab[0]); }
	
	jComplexPairAsVector256 conj(void) const { return jComplexPairAsVector256(_mm256_xor_pd(__ab, (__m256d) { 0.0, -0.0, 0.0, -0.0 })); }

	jComplexPairAsVector256& operator += (const jComplexPairAsVector256 &n) { __ab = _mm256_add_pd(__ab, n.ab()); return *this; }
	jComplexPairAsVector256 operator + (const jComplexPairAsVector256 &n) const { return jComplexPairAsVector256(*this) += n; }
	jComplexPairAsVector256& operator -= (const jComplexPairAsVector256 &n) { __ab = _mm256_sub_pd(__ab, n.ab()); return *this; }
	jComplexPairAsVector256 operator - (const jComplexPairAsVector256 &n) const { return jComplexPairAsVector256(*this) -= n; }
	jComplexPairAsVector256& operator *= (const jComplexPairAsVector256 &n)
	{
		// Multiply 'this' with n (for a and b separately)
		
		// There is probably a further optimization (at the expense of code complexity!) that can be done for the translation matrix inner loop
		// Because the coeffs for -m are almost identical to those for m (aside from a minus sign), we can avoid having to repeat several of
		// the permutations. That should shorten the critical path of the code
		
		/*		A,B * ab
					A,B * aa
					A',B' * bb
					  sign change -+-+
					[add]
				B,A * ab
					B,A * aa
					B',A' * bb
					  sign change -+-+
					[add]
				A,-B * cd
					A,-B * cc				[no sign change applied at this stage]
					A',-B' * dd				[no sign change applied at this stage. Already got A',B']
					  sign change -+-+		[no sign change applied at this stage]
					[add]					[after loop, negate second term of summation]
					
				-B,A * cd
		
				**** negation of real part of temp2 could be promoted (since swapping A and B doesn't make any difference to it - the compiler doesnt know that		
				Its on the critical path, too...
		*/
		
		
#if 0
		__m256d rere_imim = _mm256_mul_pd(__ab, n.ab());				// { x1.real() * y1.real(), x1.imag() * y1.imag(), x2.real() * y2.real(), x2.imag() * y2.imag() }
		__m256d reim_imre = _mm256_mul_pd(__ab, n.GetSwappedReIm().ab());	// { x1.real() * y1.imag(), x1.imag() * y1.real(), x2.real() * y2.imag(), x2.imag() * y2.real() }
		__m256d resultRealPart = _mm256_hsub_pd(rere_imim, rere_imim);		
		__m256d resultImagPart = _mm256_hadd_pd(reim_imre, reim_imre);
		__ab = __builtin_shufflevector(resultRealPart, resultImagPart, 0, 5, 2, 7);
#else
		__m256d duplicateReal = _mm256_permute_pd(n.ab(), 0);//__builtin_shufflevector(n.ab(), n.ab(), 0, 0, 2, 2);
		__m256d duplicateImag = _mm256_permute_pd(n.ab(), 15);//__builtin_shufflevector(n.ab(), n.ab(), 1, 1, 3, 3);
		__m256d swapped = __builtin_shufflevector(__ab, __ab, 1, 0, 3, 2);
		swapped = _mm256_xor_pd(swapped, (__m256d) { -0.0, 0.0, -0.0, 0.0 });		// Negate real part of temp2
		__m256d temp1 = _mm256_mul_pd(__ab, duplicateReal);
		__m256d temp2 = _mm256_mul_pd(swapped, duplicateImag);
//		temp2 = _mm256_xor_pd(temp2, (__m256d) { -0.0, 0.0, -0.0, 0.0 });		// Negate real part of temp2
//		temp2 = _mm256_mul_pd(temp2, (__m256d) { -1.0, 1.0, -1.0, 1.0 });		// Negate real part of temp2
		__ab = _mm256_add_pd(temp1, temp2);
#endif
		return *this;
	}
	jComplexPairAsVector256 operator * (const jComplexPairAsVector256 &n) const { return jComplexPairAsVector256(*this) *= n; }
	jComplexPairAsVector256 GetMulWithConjY(const jComplexPairAsVector256 &y) const
	{
		// Return the pairwise product of 'this' with the complex conjugate of y
#if 0
		__m256d rere_imim = _mm256_mul_pd(__ab, y.ab());				// { x1.real() * y1.real(), x1.imag() * y1.imag(), x2.real() * y2.real(), x2.imag() * y2.imag() }
		__m256d imre_reim = _mm256_mul_pd(GetSwappedReIm().ab(), y.ab());	// { x1.imag() * y1.real(), x1.real() * y1.imag(), x2.imag() * y2.real(), x2.real() * y2.imag() }
		__m256d resultRealPart = _mm256_hadd_pd(rere_imim, rere_imim);		// x1.re * y1.re + x1.im * y1.im
		__m256d resultImagPart = _mm256_hsub_pd(imre_reim, imre_reim);		// x1.im * y1.re - x1.re * y1.im
		__m256d result = __builtin_shufflevector(resultRealPart, resultImagPart, 0, 5, 2, 7);
		return jComplexPairAsVector256(result);
#else
		__m256d duplicateReal = _mm256_permute_pd(y.ab(), 0);//__builtin_shufflevector(n.ab(), n.ab(), 0, 0, 2, 2);
		__m256d duplicateImag = _mm256_permute_pd(y.ab(), 15);//__builtin_shufflevector(n.ab(), n.ab(), 1, 1, 3, 3);
		__m256d swapped = __builtin_shufflevector(__ab, __ab, 1, 0, 3, 2);
		swapped = _mm256_xor_pd(swapped, (__m256d) { 0.0, -0.0, 0.0, -0.0 });		// Negate imag part of temp2
		__m256d temp1 = _mm256_mul_pd(__ab, duplicateReal);
		__m256d temp2 = _mm256_mul_pd(swapped, duplicateImag);
		//		temp2 = _mm256_xor_pd(temp2, (__m256d) { -0.0, 0.0, -0.0, 0.0 });		// Negate real part of temp2
		//		temp2 = _mm256_mul_pd(temp2, (__m256d) { -1.0, 1.0, -1.0, 1.0 });		// Negate real part of temp2
		return jComplexPairAsVector256(_mm256_add_pd(temp1, temp2));
#endif
	}
	jComplexPairAsVector256& operator /= (const jComplexPairAsVector256 &n)
	{
		jComplexPairAsVector256 conjMul = GetMulWithConjY(n);
		__m256d normIntermediate = _mm256_mul_pd(n.ab(), n.ab());
		__m256d nNorm = _mm256_hadd_pd(normIntermediate, normIntermediate);
		__ab = _mm256_div_pd(conjMul.ab(), nNorm);
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

	jComplexPairAsVector256& operator += (const double &n) { __ab = _mm256_add_pd(__ab, (__m256d){n, 0, n, 0}); return *this; }
	jComplexPairAsVector256 operator + (const double &n) const { return jComplexPairAsVector256(*this) += n; }
	jComplexPairAsVector256& operator -= (const double &n) { __ab = _mm256_sub_pd(__ab, (__m256d){n, 0, n, 0}); return *this; }
	jComplexPairAsVector256 operator - (const double &n) const { return jComplexPairAsVector256(*this) -= n; }
	jComplexPairAsVector256& operator *= (double n) { __ab = _mm256_mul_pd(__ab, (__m256d){n, n, n, n}); return *this; }
	jComplexPairAsVector256 operator * (double n) const { return jComplexPairAsVector256(*this) *= n; }
	jComplexPairAsVector256& operator /= (double n) { __ab = _mm256_div_pd(__ab, (__m256d){n, n, n, n}); return *this; }
	jComplexPairAsVector256 operator / (double n) const { return jComplexPairAsVector256(*this) /= n; }

	double SumReal(void) const
	{
		// Return the sum of the real parts of the two numbers
		__m128d high = _mm256_extractf128_pd(__ab, 1), low = _mm256_castpd256_pd128(__ab);
		return _mm_cvtsd_f64(_mm_add_pd(high, low));
	}
	double SumImag(void) const
	{
		// Return the sum of the imaginary parts of the two numbers
		__m128d high = _mm256_extractf128_pd(__ab, 1), low = _mm256_castpd256_pd128(__ab);
		__m128d sum = _mm_add_pd(high, low);
		return sum[1];
	}
	double SumNorm(void) const
	{
		// Return the sum of the norms of the two numbers
		// NOTE: as per complex::norm, we use norm in the sense of the *squared* magnitude,
		// which is not the standard mathematical definition, but is what is used in complex::norm.
		__m256d normIntermediate = _mm256_mul_pd(__ab, __ab);
		__m256d norm = _mm256_hadd_pd(normIntermediate, normIntermediate);
		__m128d high = _mm256_extractf128_pd(norm, 1), low = _mm256_castpd256_pd128(norm);
		return _mm_cvtsd_f64(_mm_add_pd(high, low));
	}
	jComplex SumAcross(void) const
	{
		// Return the sum of the two values as a single complex number
		// TODO: might it be possible to come up with a shorter instruction sequence here? (and potentially the functions above?)
		__m128d high = _mm256_extractf128_pd(__ab, 1), low = _mm256_castpd256_pd128(__ab);
		__m128d result = _mm_add_pd(low, high);
		return jComplex(vLower(result), vUpper(result));
	}
	jComplexPairAsVector256 Negated(double ar, double ai, double br, double bi) const { return jComplexPairAsVector256(_mm256_mul_pd(__ab, (__m256d) { ar, ai, br, bi })); }
    jComplexPairAsVector256 NegatedUsingXORWith(jComplexPairAsVector256 neg) const { return jComplexPairAsVector256(_mm256_xor_pd(__ab, neg.ab())); }   // Expects a vector containing either 0.0 or -0.0 entries. Uses the XOR operator to negate self according to that pattern
	jComplexPairAsVector256 GetNegative(void) const { return jComplexPairAsVector256(_mm256_xor_pd(__ab, (__m256d) { -0.0, -0.0, -0.0, -0.0 })); }		// Return { -a, -b }
    jComplexPairAsVector256 GetNegativeOfFirstOnly(void) const { return jComplexPairAsVector256(_mm256_xor_pd(__ab, (__m256d) { -0.0, -0.0, 0.0, 0.0 })); }		// Return { -a, b }
	jComplexPairAsVector256 GetNegativeOfSecondOnly(void) const { return jComplexPairAsVector256(_mm256_xor_pd(__ab, (__m256d) { 0.0, 0.0, -0.0, -0.0 })); }		// Return { a, -b }
	jComplexPairAsVector256 GetSwappedPairsAndReIm(void) const { return jComplexPairAsVector256(__builtin_shufflevector(__ab, __ab, 3, 2, 1, 0)); }	// Return { b, a } given { a, b }
	jComplexPairAsVector256 GetSwappedPairs(void) const { return jComplexPairAsVector256(__builtin_shufflevector(__ab, __ab, 2, 3, 0, 1)); }	// Return { b, a } given { a, b }
	jComplexPairAsVector256 GetSwappedReIm(void) const { return jComplexPairAsVector256(__builtin_shufflevector(__ab, __ab, 1, 0, 3, 2)); }	// Return { im(a), re(a), im(b), re(b) } given { a, b }

	void Print(const char *suffix = "") const
	{
		printf("{");
		::Print(a());
		printf(", ");
		::Print(b());
		printf("}%s", suffix);
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

inline jComplexPairAsVector256 re_part(const jComplexPairAsVector256 &x) { return jComplexPairAsVector256(_mm256_and_pd(x.ab(), (__v8si) { -1, -1, 0, 0, -1, -1, 0, 0 })); }
inline jComplexPairAsVector256 im_part(const jComplexPairAsVector256 &x) { return jComplexPairAsVector256(_mm256_and_pd(x.ab(), (__v8si) { 0, 0, -1, -1, 0, 0, -1, -1 })); }
inline double SumReal(const jComplexPairAsVector256 &x) { return x.SumReal(); }
inline double SumNorm(const jComplexPairAsVector256 &x) { return x.SumNorm(); }
inline jComplex SumAcross(const jComplexPairAsVector256 &x) { return x.SumAcross(); }
inline jComplexPairAsVector256 MulXConjY(const jComplexPairAsVector256 &x, const jComplexPairAsVector256 &y) { return x.GetMulWithConjY(y); }

inline jComplexPairAsVector256 SwapPairs(const jComplexPairAsVector256 &x) { return jComplexPairAsVector256(x.GetSwappedPairs()); }

#endif
