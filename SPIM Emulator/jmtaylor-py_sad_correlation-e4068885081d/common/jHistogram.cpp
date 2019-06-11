/*
 *  jHistogram.cpp
 *
 *  Copyright 2011-2015 Jonathan Taylor. All rights reserved.
 *
 *  C++ class to maintain a histogram based on datapoints that are added one by one
 *  Caller can query contents of any particular histogram bin.
 *
 */

#include "jHistogram.h"
#include <algorithm>

jHistogram::jHistogram(double fbs/*first bin starts at*/, double bw/*bin width*/, int numBins, bool io/*include outliers in end bins*/)
{
	includeOutliers = io;
	SetHistogramParams(fbs, bw, numBins);
	numMissed = 0;
	missedVal = INT_MIN;
}

void jHistogram::SetHistogramParams(double fbs/*first bin starts at*/, double bw/*bin width*/, int numBins, bool reset)
{
	if (includeOutliers)
		CHECK(reset == true);
		
	if (!reset)
	{
		// In some circumstances we can retain the existing data while extending the histogram range
		double newFirstBinDouble = (fbs - firstBinStart) / bw;
		int newFirstBin = int(newFirstBinDouble);
		if ((bw == binWidth) &&
			(newFirstBinDouble - newFirstBin == 0.0))
		{
			if (newFirstBin < 0)
				histogram.insert(histogram.begin(), -newFirstBin, 0);
			else if (newFirstBin > 0)
				histogram.erase(histogram.begin(), histogram.begin() + newFirstBin);
			if (numBins < int(histogram.size()))
				histogram.erase(histogram.begin() + numBins);
			else
				histogram.insert(histogram.end(), numBins - histogram.size(), 0);
			firstBinStart = fbs;
			binWidth = bw;
			return;
		}
		else
		{
			// If we are not asked to reset then the caller should have ensured these conditions were met to be able to retain existing data
			CHECK(false);
			// Fall through to reset code
		}
	}
	
	// Resize and reset
	numMissed = 0;
	histogram = std::vector<int>(numBins);
	firstBinStart = fbs;
    fbsInt = fbs;
	binWidth = bw;
}

double jHistogram::MaxVal(void) const
{
	// Greatest value in the histogram
	return *std::max_element(histogram.begin(), histogram.end());
}

void jHistogram::Print(void)
{
	for (size_t i = 0; i < histogram.size(); i++)
		printf("%zd\t%le\t%d\n", i, firstBinStart + i * binWidth, histogram[i]);
}
