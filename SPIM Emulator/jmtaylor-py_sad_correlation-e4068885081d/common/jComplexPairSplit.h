/*
 *	jComplexPairSplit.h
 *
 *  Copyright 2011-2015 Jonathan Taylor. All rights reserved.
 *
 *  Class representing two complex numbers, storing the values as two jComplex objects
 *	This is a generic reference implementation, but it is not particularly high performance - 
 *  if SSE is available then jComplexPairAsVectoris a much better choice
 */

#ifndef __JCOMPLEX_PAIR_SPLIT_H__
#define __JCOMPLEX_PAIR_SPLIT_H__

#include "jComplex.h"

template <class ComplexType, class DoubleType> class jComplexPairSplitT
{
	// This class represents two complex numbers
	// This implementation just acts as a container for two instances of jComplex
  protected:
	ComplexType __a, __b;
	
  public:

	jComplexPairSplitT() : __a(), __b() { }
	static jComplexPairSplitT<ComplexType, DoubleType> zero(void) { return jComplexPairSplitT<ComplexType, DoubleType>(DoubleType(0), DoubleType(0)); }

	ComplexType a(void) const { return __a; }
	ComplexType b(void) const { return __b; }

	explicit jComplexPairSplitT(const ComplexType &inZ) : __a(inZ), __b(inZ) { }
	jComplexPairSplitT(const jComplexPairSplitT &inAB) : __a(inAB.a()), __b(inAB.b()) { }
	jComplexPairSplitT(ComplexType inA, ComplexType inB) : __a(inA), __b(inB) { }
	
	void SetA(ComplexType inA) { __a = inA; }
	void SetB(ComplexType inB) { __b = inB; }
	
	jComplexPairSplitT& operator += (const jComplexPairSplitT &n) { __a += n.a(); __b += n.b(); return *this; }
	jComplexPairSplitT operator + (const jComplexPairSplitT &n) const { return jComplexPairSplitT(*this) += n; }
	jComplexPairSplitT& operator -= (const jComplexPairSplitT &n) { __a -= n.a(); __b -= n.b(); return *this; }
	jComplexPairSplitT operator - (const jComplexPairSplitT &n) const { return jComplexPairSplitT(*this) -= n; }
	jComplexPairSplitT& operator *= (const jComplexPairSplitT &n) { __a *= n.a(); __b *= n.b(); return *this; }
	jComplexPairSplitT operator * (const jComplexPairSplitT &n) const { return jComplexPairSplitT(*this) *= n; }
	jComplexPairSplitT& operator /= (const jComplexPairSplitT &n) { __a /= n.a(); __b /= n.b(); return *this; }
	jComplexPairSplitT operator / (const jComplexPairSplitT &n) const { return jComplexPairSplitT(*this) /= n; }

	jComplexPairSplitT& operator += (const ComplexType &n) { __a += n; __b += n; return *this; }
	jComplexPairSplitT operator + (const ComplexType &n) const { return jComplexPairSplitT(*this) += n; }
	jComplexPairSplitT& operator -= (const ComplexType &n) { __a -= n; __b -= n; return *this; }
	jComplexPairSplitT operator - (const ComplexType &n) const { return jComplexPairSplitT(*this) -= n; }
	jComplexPairSplitT& operator *= (const ComplexType &n) { __a *= n; __b *= n; return *this; }
	jComplexPairSplitT operator * (const ComplexType &n) const { return jComplexPairSplitT(*this) *= n; }
	jComplexPairSplitT& operator /= (const ComplexType &n) { __a /= n; __b /= n; return *this; }
	jComplexPairSplitT operator / (const ComplexType &n) const { return jComplexPairSplitT(*this) /= n; }

	jComplexPairSplitT& operator += (const DoubleType &n) { __a += n; __b += n; return *this; }
	jComplexPairSplitT operator + (const DoubleType &n) const { return jComplexPairSplitT(*this) += n; }
	jComplexPairSplitT& operator -= (const DoubleType &n) { __a -= n; __b -= n; return *this; }
	jComplexPairSplitT operator - (const DoubleType &n) const { return jComplexPairSplitT(*this) -= n; }
	jComplexPairSplitT& operator *= (DoubleType n) { __a *= n; __b *= n; return *this; }
	jComplexPairSplitT operator * (DoubleType n) const { return jComplexPairSplitT(*this) *= n; }
	jComplexPairSplitT& operator /= (DoubleType n) { __a /= n; __b /= n; return *this; }
	jComplexPairSplitT operator / (DoubleType n) const { return jComplexPairSplitT(*this) /= n; }
	
	jComplexPairSplitT conj(void) const { return jComplexPairSplitT(::conj(__a), ::conj(__b)); }

	DoubleType SumReal(void) const { return real(__a) + real(__b); }
	DoubleType SumNorm(void) const { return norm(__a) + norm(__b); }
	ComplexType SumAcross(void) const { return __a + __b; }
	jComplexPairSplitT GetNegative(void) const { return jComplexPairSplitT(-__a, -__b); }
	jComplexPairSplitT GetSwappedPairs(void) const { return jComplexPairSplitT(__b, __a); }
	jComplexPairSplitT GetMulWithConjY(const jComplexPairSplitT &y) const { return jComplexPairSplitT(__a * ::conj(y.a()), __b * ::conj(y.b())); }
	jComplexPairSplitT NegatedUsingXORWith(jComplexPairSplitT neg) const
	{
		// Expects a vector containing either 0.0 or -0.0 entries.
		// Of course, we cannot actually use XOR in this implementation (the call is designed as a hint for vectorized implementations)
		return jComplexPairSplitT(ComplexType(__a.real() * sign(neg.a().real()),
										     __a.imag() * sign(neg.a().imag())),
								 ComplexType(__b.real() * sign(neg.b().real()),
										     __b.imag() * sign(neg.b().imag())));
	}
    jComplexPairSplitT GetNegativeOfFirstOnly(void) const { return jComplexPairSplitT(-__a, __b); }
	jComplexPairSplitT GetNegativeOfSecondOnly(void) const { return jComplexPairSplitT(__a, -__b); }

	void Print(const char *suffix = "") const
	{
		printf("{");
		::Print(a());
		printf(", ");
		::Print(b());
		printf("}%s", suffix);
	}
};

template<class ComplexType, class DoubleType> inline jComplexPairSplitT<ComplexType, DoubleType> operator*(const DoubleType l, const jComplexPairSplitT<ComplexType, DoubleType> &r)
{
	return r * l;
}

template<class ComplexType, class DoubleType> inline jComplexPairSplitT<ComplexType, DoubleType> operator*(const ComplexType l, const jComplexPairSplitT<ComplexType, DoubleType> &r)
{
	return r * l;
}

template<class ComplexType, class DoubleType> inline jComplexPairSplitT<ComplexType, DoubleType> operator/(const DoubleType l, const jComplexPairSplitT<ComplexType, DoubleType> &r)
{
	return jComplexPairSplitT<ComplexType, DoubleType>(l / r.a(), l / r.b());
}

template<class ComplexType, class DoubleType> inline jComplexPairSplitT<ComplexType, DoubleType> operator/(const ComplexType l, const jComplexPairSplitT<ComplexType, DoubleType> &r)
{
	return jComplexPairSplitT<ComplexType, DoubleType>(l / r.a(), l / r.b());
}

template<class ComplexType, class DoubleType> inline jComplexPairSplitT<ComplexType, DoubleType> operator-(const DoubleType l, const jComplexPairSplitT<ComplexType, DoubleType> &r)
{
	return jComplexPairSplitT<ComplexType, DoubleType>(l - r.a(), l - r.b());
}

template<class ComplexType, class DoubleType> inline jComplexPairSplitT<ComplexType, DoubleType> operator-(const jComplexPairSplitT<ComplexType, DoubleType> &r)
{
	return r.GetNegative();
}

template<class ComplexType, class DoubleType> inline jComplexPairSplitT<ComplexType, DoubleType> operator+(const jComplexPairSplitT<ComplexType, DoubleType> &r)
{
	return r;
}

template<class ComplexType, class DoubleType> inline jComplexPairSplitT<ComplexType, DoubleType> operator+(const DoubleType l, const jComplexPairSplitT<ComplexType, DoubleType> &r)
{
	return r + l;
}

template<class ComplexType, class DoubleType> inline jComplexPairSplitT<ComplexType, DoubleType> conj(const jComplexPairSplitT<ComplexType, DoubleType> &z)
{
	return z.conj();
}

template<class ComplexType, class DoubleType> inline jComplexPairSplitT<ComplexType, DoubleType> re_part(const jComplexPairSplitT<ComplexType, DoubleType> &x) { return jComplexPairSplitT<ComplexType, DoubleType>(real(x.a()), real(x.b())); }
template<class ComplexType, class DoubleType> inline jComplexPairSplitT<ComplexType, DoubleType> im_part(const jComplexPairSplitT<ComplexType, DoubleType> &x) { return jComplexPairSplitT<ComplexType, DoubleType>(imag(x.a()), imag(x.b())); }
template<class ComplexType, class DoubleType> inline DoubleType SumReal(const jComplexPairSplitT<ComplexType, DoubleType> &x) { return x.SumReal(); }
template<class ComplexType, class DoubleType> inline DoubleType SumNorm(const jComplexPairSplitT<ComplexType, DoubleType> &x) { return x.SumNorm(); }
template<class ComplexType, class DoubleType> inline ComplexType SumAcross(const jComplexPairSplitT<ComplexType, DoubleType> &x) { return x.SumAcross(); }
template<class ComplexType, class DoubleType> inline jComplexPairSplitT<ComplexType, DoubleType> MulXConjY(const jComplexPairSplitT<ComplexType, DoubleType> &x, const jComplexPairSplitT<ComplexType, DoubleType> &y) { return x.GetMulWithConjY(y); }

template<class ComplexType, class DoubleType> inline jComplexPairSplitT<ComplexType, DoubleType> SwapPairs(const jComplexPairSplitT<ComplexType, DoubleType> &x) { return jComplexPairSplitT<ComplexType, DoubleType>(x.GetSwappedPairs()); }

#endif
