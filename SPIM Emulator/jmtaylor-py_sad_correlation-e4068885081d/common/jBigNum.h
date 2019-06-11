/*
 *	jBigNum.h
 *
 *	Copyright 2010-2015 Jonathan Taylor. All rights reserved.
 *
 *  This class represents a complex number that can be potentially very large,
 *	with operator overloading allowing instances to be operated on in standard
 *	arithmetic expressions without any special treatment.
 *	Note that not all operators are implemented - I have only implemented the ones I need.
 *
 *	When dealing with translation matrices for very large spheres we can end up with huge numbers.
 *	This class can cope with numbers up to exp(10 * LONG_MAX)
 *	The class is not as elegant as it could be. It is necessary to manually call Renormalize()
 *	after a sequence of operations which may have caused the value of _detail to become close to saturation.
 *	If Renormalize() is not called enough then we will end up with 'inf' values. That will mess up the calculation,
 *	but it will be clear from the end result that a problem has occurred due to the presence of infinities and/or NaNs
 *	in the results.
 *
 *	NOTE: I introduced this class some time ago, and since then I have implemented the jreal wrapper around 
 *	the GMP library. In many cases that may be a preferable alternative, but jBigNum will be significantly faster
 *	(albeit more unwieldy and limited in features) in cases where the problem is simply that there are large exponents,
 *	rather than an actual requirement for higher precision.
 *	It is mostly here just to support legacy code, though.
 *
 */

#ifndef __JBIGNUM_H__
#define __JBIGNUM_H__

#include <stdio.h>
#include <complex>
#include "jComplex.h"
#include "float.h"

class jBigNum
{
  public:
	static const long kBigNumMaxExponentInTable = 10 /* Any larger overflows a 'double' */;
	static const long kBigNumExponentPowerOfE = 64;

  protected:
	jComplexR	_detail;
	long		_exponent;
	
	static jreal expTable[kBigNumMaxExponentInTable + 1];
	static jreal invExpTable[kBigNumMaxExponentInTable + 1];
	static jreal logExponent;

  public:

	jBigNum() { _detail = jreal(0); _exponent = 0; }
	explicit jBigNum(jreal inVal) { _detail = inVal; _exponent = 0; }
	explicit jBigNum(jComplexR inVal) { _detail = inVal; _exponent = 0; }
	jBigNum(jComplexR inVal, long inExponent) { _detail = inVal; _exponent = inExponent; }
	static jBigNum zero(void) { return jBigNum(jreal(0), 0); }
	
	static void InitBigNum(void);
	
	long exponent(void) const { return _exponent; }
	jComplexR detail(void) const { return _detail; }
	
	bool	FitsInDouble(void) const { return (abs((int)_exponent) < (DBL_MAX_10_EXP - 10)); }

	jBigNum& operator += (const jBigNum &n)
	{
		if (_exponent == n.exponent()) { _detail += n.detail(); }
		else if (_exponent < n.exponent())
		{
			if (n.exponent() - _exponent < kBigNumMaxExponentInTable)
				_detail = invExpTable[n.exponent() - _exponent] * _detail + n.detail(); _exponent = n.exponent();
		}
		else
		{
			if (_exponent - n.exponent() < kBigNumMaxExponentInTable)
				_detail += invExpTable[_exponent - n.exponent()] * n.detail();
		}
		return *this;
	}
	jBigNum operator + (const jBigNum &n) const { return jBigNum(*this) += n; }
	
	jBigNum& operator += (const jComplexR &n)
	{
		return operator += (jBigNum(n, 0));
	}
	jBigNum operator + (const jComplexR &n) const { return jBigNum(*this) += n; }
	jBigNum& operator += (const jreal n)
	{
		return operator +=(jBigNum(n, 0));
	}
	jBigNum operator + (const jreal n) const { return jBigNum(*this) += n; }
	
	jBigNum& operator -= (const jBigNum &n)
	{
		if (_exponent == n.exponent()) { _detail -= n.detail(); }
		else if (_exponent < n.exponent())
		{
			if (n.exponent() - _exponent < kBigNumMaxExponentInTable)
				_detail = invExpTable[n.exponent() - _exponent] * _detail - n.detail(); _exponent = n.exponent();
		}
		else
		{
			if (_exponent - n.exponent() < kBigNumMaxExponentInTable)
				_detail -= invExpTable[_exponent - n.exponent()] * n.detail();
		}
		return *this;
	}
	jBigNum operator - (const jBigNum &n) const { return jBigNum(*this) -= n; }
	
	jBigNum& operator -= (const jComplexR &n)
	{
		return operator -= (jBigNum(n, 0));
	}
	jBigNum operator - (const jComplexR &n) const { return jBigNum(*this) -= n; }
	jBigNum& operator -= (const jreal n)
	{
		return operator -= (jBigNum(n, 0));
	}
	jBigNum operator - (const jreal n) const { return jBigNum(*this) -= n; }

	jBigNum& operator *= (const jreal n) { _detail *= n; return *this; }
	jBigNum operator * (const jreal n) const { return jBigNum(*this) *= n; }
	jBigNum& operator *= (const jComplexR &n) { _detail *= n; return *this; }
	jBigNum operator * (const jComplexR &n) const { return jBigNum(*this) *= n; }
	jBigNum& operator *= (const jBigNum &n)
	{
		_detail *= n.detail();
		_exponent += n.exponent();
		return *this;
	}
	jBigNum operator * (const jBigNum &n) const { return jBigNum(*this) *= n; }
	
	jBigNum& operator /= (const jBigNum &n)
	{
		_detail /= n.detail();
		_exponent -= n.exponent();
		return *this;
	}
	jBigNum operator / (const jBigNum &n) const { return jBigNum(*this) /= n; }
	jBigNum& operator /= (jComplexR n) { _detail /= n; return *this; }
	jBigNum operator / (jComplexR n) const { return jBigNum(*this) /= n; }
	
	jBigNum conj(void) const { return jBigNum(std::conj(_detail), _exponent); }
	jComplexR to_jcomplex(void) const
	{
		if (_exponent >= 0)
		{
			if (_exponent <= kBigNumMaxExponentInTable)
				return _detail * expTable[_exponent];
			else
				return jreal_consts::dbl_max();
		}
		else
		{
			if (_exponent >= -kBigNumMaxExponentInTable)
				return _detail * invExpTable[-_exponent];
			else
				return jreal_consts::dbl_min();
		}
	}	

	jreal abs_r(void) const
	{
		return abs(to_jcomplex());
	}
	
	double abs_d(void) const
	{
		return AllowPrecisionLossReadingValue(abs(to_jcomplex()));
	}

	jreal logAbs(void) const
	{
		return log(abs(_detail)) + logExponent * _exponent;
	}
		
	void Renormalize(void)
	{
		// Check it's not NaN
		ALWAYS_ASSERT(_detail == _detail);
		
		// Reduce the detail part if it's too large
		long i;
//		jreal absDetail = abs(_detail);
		jreal absDetail = sqrt(SQUARE(_detail.real()) + SQUARE(_detail.imag()));
		for (i = 0; absDetail > expTable[i+1]; i++)
		{
			// Check for overflow
			ALWAYS_ASSERT(i < kBigNumMaxExponentInTable);
		}
		_exponent += i;
		_detail *= invExpTable[i];
		absDetail *= invExpTable[i];

		// Increase the detail part if it's too large
		if (absDetail != jreal(0))
		{
			for (i = 0; absDetail < invExpTable[i+1]; i++)
			{
				// Check for underflow
				ALWAYS_ASSERT(i < kBigNumMaxExponentInTable);
			}
		}
		_exponent -= i;
		_detail *= expTable[i];
	}
};

inline jBigNum operator*(const jreal l, const jBigNum &r)
{
	return r * l;
}

inline jBigNum operator*(const jComplexR l, const jBigNum &r)
{
	return r * l;
}

inline jBigNum operator-(const jBigNum &r)
{
	return jBigNum(-r.detail(), r.exponent());
}

inline jBigNum operator+(const jBigNum &r)
{
	return r;
}

inline jBigNum operator-(const jreal l, const jBigNum &r)
{
	return (-r) + l;
}

inline jBigNum operator-(const jComplexR l, const jBigNum &r)
{
	return (-r) + l;
}

inline jBigNum operator+(const jreal l, const jBigNum &r)
{
	return r + l;
}

inline jBigNum operator+(const jComplexR l, const jBigNum &r)
{
	return r + l;
}

inline jBigNum operator/(const jreal l, const jBigNum &r)
{
	return jBigNum(l / r.detail(), -r.exponent());
}

inline jBigNum operator/(const jComplexR l, const jBigNum &r)
{
	return jBigNum(l / r.detail(), -r.exponent());
}

inline jreal abs_r(const jBigNum &n)
{
	return n.abs_r();
}

inline double abs_d(const jBigNum &n)
{
	return n.abs_d();
}

inline jreal logAbs(const jBigNum &n)
{
	return n.logAbs();
}

inline jBigNum conj(const jBigNum &n)
{
	return n.conj();
}

inline bool is_nan(const jBigNum &z)
{
	return (z.detail() != z.detail());
}


inline bool operator==(const jBigNum &a, const jBigNum &b)
{
	jBigNum a1 = a, b1 = b;
	a1.Renormalize();
	b1.Renormalize();
	return ((a1.exponent() == b1.exponent()) && (a1.detail() == b1.detail()));
}

inline bool operator==(const jBigNum &a, const jComplexR &b)
{
	return (a.to_jcomplex() == b);
}

void Print(jBigNum z, const char *suffix = "");
void PrintDecimal(jBigNum z);

bool CheckAgreement(jBigNum val1, jComplex val2, double relError, double absError, bool printOnDisagreement = true, double *amount = NULL);
bool CheckAgreement(jComplexR val1, jBigNum val2, double relError, double absError, bool printOnDisagreement = true, double *amount = NULL);

#endif
