/*
 *	jBigNum.cpp
 *
 *	Copyright 2010-2015 Jonathan Taylor. All rights reserved.
 *
 *  Utility functions associated with jBigNum class (see header file)
 */
#include "jBigNum.h"

jreal jBigNum::expTable[kBigNumMaxExponentInTable + 1];
jreal jBigNum::invExpTable[kBigNumMaxExponentInTable + 1];
jreal jBigNum::logExponent;

void Print(jBigNum z, const char *suffix)
{
	Print(z.detail());
	printf(" x e^%ld%s", z.exponent() * jBigNum::kBigNumExponentPowerOfE, suffix);
}

void MakeScientificNotation(jreal &x, long &exponent)
{
	while (fabs(x) >= jreal(10))
	{
		x *= jreal(0.1);
		exponent++;
	}
	while (fabs(x) < jreal(1))
	{
		x *= jreal(10);
		exponent--;
	}
}

void PrintOneComponent(jreal detail, long exponent)
{
	jreal detailPart = detail;
	long decimalExponent = 0;
	long i;

	if (detail == jreal(0))
	{
		printf("0.000000e+00");
		return;
	}
	MakeScientificNotation(detailPart, decimalExponent);
	if (exponent >= 0)
	{
		for (i = 0; i < exponent; i++)
		{
			detailPart *= exp(jreal(jBigNum::kBigNumExponentPowerOfE));
			MakeScientificNotation(detailPart, decimalExponent);
		}
	}
	else
	{
		for (i = exponent; i < 0; i++)
		{
			detailPart /= exp(jreal(jBigNum::kBigNumExponentPowerOfE));
			MakeScientificNotation(detailPart, decimalExponent);
		}
	}
	printf("%.6lfe%+.02ld", AllowPrecisionLossReadingValue(detailPart), decimalExponent);
}	
	
void PrintDecimal(jBigNum z)
{
	printf("{");
	PrintOneComponent(real(z.detail()), z.exponent());
	printf(", ");
	PrintOneComponent(imag(z.detail()), z.exponent());
	printf("}");
}


void jBigNum::InitBigNum(void)
{
	for (long i = 0; i <= kBigNumMaxExponentInTable; i++)
	{
		expTable[i] = exp(jreal(kBigNumExponentPowerOfE * i));
		invExpTable[i] = 1.0 / expTable[i];
		logExponent = log(exp(jreal(kBigNumExponentPowerOfE)));
	}
}

bool CheckAgreement(jBigNum val1, jComplex val2, double relError, double absError, bool printOnDisagreement, double *amount)
{
	if (!val1.FitsInDouble())
	{
		if (printOnDisagreement)
		{
			printf("DISAGREEMENT: ");
			Print(val1);
			printf(" and ");
			Print(val2);
		}
		return false;
	}
	return CheckAgreement(val1.to_jcomplex(), val2, relError, absError, printOnDisagreement, amount);
}

bool CheckAgreement(jComplex val1, jBigNum val2, double relError, double absError, bool printOnDisagreement, double *amount)
{
	if (!val2.FitsInDouble())
	{
		if (printOnDisagreement)
		{
			printf("DISAGREEMENT: ");
			Print(val1);
			printf(" and ");
			Print(val2);
		}
		return false;
	}
	return CheckAgreement(val1, val2.to_jcomplex(), relError, absError, printOnDisagreement, amount);
}
