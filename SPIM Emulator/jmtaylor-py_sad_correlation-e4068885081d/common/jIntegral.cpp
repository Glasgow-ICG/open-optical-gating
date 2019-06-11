//
//  jIntegral.cpp
//
//	Copyright 2010-2015 Jonathan Taylor. All rights reserved.
//
//	A few specific implementations for jIntegral.h
//

#include "jIntegral.h"
#include "jComplex.h"
#include "jCoord.h"

namespace jIntegralPrivate
{
	template<> double GetZeroSum<double>(void) { return 0.0; }
	template<> jComplex GetZeroSum<jComplex>(void) { return 0.0; }
	template<> coord3 GetZeroSum<coord3>(void) { return coord3(0.0, 0.0, 0.0); }
	template<> coordC3 GetZeroSum<coordC3>(void) { return coordC3(0.0, 0.0, 0.0); }

	bool CompareByAbsolute(double a, double b)
	{
		return (fabs(a) < fabs(b));
	}

	template<> double SumResultList<double>(std::vector<double> &resultList)
	{
        /*  This is probably over-engineering, but this is intended to be a way of ensuring
            maximum accuracy by adding the smallest values together first, thereby minimizing rounding errors. 
            Obviously this has significant performance implications, in terms of memory used and the time required for the sorting!*/
		double thisResult = 0.0;
		sort(resultList.begin(), resultList.end(), CompareByAbsolute);
		for (unsigned int j = 0; j < resultList.size(); j++)
			thisResult += resultList[j];
		return thisResult;
	}

	// This can be implemented for these types (using 2, 3 and 6 sorts respectively), but I'm not going to bother unless I need it
	template<> jComplex SumResultList<jComplex>(jComplexVector &resultList) { ALWAYS_ASSERT(0); return GetZeroSum<jComplex>(); }
    template<> coord3 SumResultList<coord3>(std::vector<coord3> &resultList) { ALWAYS_ASSERT(0); return GetZeroSum<coord3>(); }
    template<> coordC3 SumResultList<coordC3>(std::vector<coordC3> &resultList) { ALWAYS_ASSERT(0); return GetZeroSum<coordC3>(); }
}

