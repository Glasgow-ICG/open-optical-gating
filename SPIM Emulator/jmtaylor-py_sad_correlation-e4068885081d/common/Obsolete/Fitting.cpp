/*
 *  Fitting.cpp
 *
 *  Created by Jonathan Taylor on 08/09/2011.
 *  Copyright 2011 Durham University. All rights reserved.
 *
 */

#include "Fitting.h"
#include <sstream>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_multifit.h>
#include "jAssert.h"

void DrawFitAndPlot(std::vector<double> x, std::vector<double> y, double params[3], const char *gnuplotTitle)
{
	std::ostringstream gnuplotString;
	// There is some hideous escaping here because we need to get an escaped single quote to
	// appear in str (but also need to do some escaping to protect those characters in this C++ source file!)
	gnuplotString << "echo 'plot '\"'\"'-'\"'\"' using 1:2 with lines ti '\"'\"'" << gnuplotTitle << "'\"'\"'\n";
	ALWAYS_ASSERT(x.size() == y.size());

	for (size_t i = 0; i < x.size(); i++)
		gnuplotString << x[i] << " " << y[i] << "\n";
	gnuplotString << "\n";
	for (size_t i = 0; i < x.size(); i++)
		gnuplotString << x[i] << " " << params[0] + params[1] * x[i] + params[2] * x[i] * x[i] << "\n";

	gnuplotString << "e' | /usr/local/bin/gnuplot";
	system(gnuplotString.str().c_str());
}

void QuadraticFit(std::vector<double> x, std::vector<double> y, double outParams[3], const char *gnuplotTitle)
{
	double xi, yi, ei, chisq;
	gsl_matrix *X, *cov;
	gsl_vector *yv, *w, *c;
	
	ALWAYS_ASSERT(x.size() == y.size());
	size_t n = x.size();

	X = gsl_matrix_alloc (n, 3);
	yv = gsl_vector_alloc (n);
	w = gsl_vector_alloc (n);

	c = gsl_vector_alloc (3);
	cov = gsl_matrix_alloc (3, 3);

	for (size_t i = 0; i < n; i++)
	{
		xi = x[i];
		yi = y[i];
		ei = 1e-10;
		gsl_matrix_set (X, i, 0, 1.0);
		gsl_matrix_set (X, i, 1, xi);
		gsl_matrix_set (X, i, 2, xi*xi);
		gsl_vector_set (yv, i, yi);
		gsl_vector_set (w, i, 1.0/(ei*ei));
	}

	gsl_multifit_linear_workspace * work = gsl_multifit_linear_alloc (n, 3);
	gsl_multifit_wlinear (X, w, yv, c, cov, &chisq, work);
	gsl_multifit_linear_free (work);
     
	outParams[0] = gsl_vector_get(c, 0);
	outParams[1] = gsl_vector_get(c, 1);
	outParams[2] = gsl_vector_get(c, 2);
	printf ("# best fit: Y = %g + %g X + %g X^2\n", outParams[0], outParams[1], outParams[2]);

	if (gnuplotTitle != NULL)
		DrawFitAndPlot(x, y, outParams, gnuplotTitle);
	
	gsl_matrix_free (X);
	gsl_vector_free (yv);
	gsl_vector_free (w);
	gsl_vector_free (c);
	gsl_matrix_free (cov);
}

void LinearFit(std::vector<double> x, std::vector<double> y, double outParams[2], const char *gnuplotTitle)
{
	double xi, yi, ei, chisq;
	gsl_matrix *X, *cov;
	gsl_vector *yv, *w, *c;
	
	ALWAYS_ASSERT(x.size() == y.size());
	size_t n = x.size();

	X = gsl_matrix_alloc (n, 2);
	yv = gsl_vector_alloc (n);
	w = gsl_vector_alloc (n);

	c = gsl_vector_alloc (2);
	cov = gsl_matrix_alloc (2, 2);

	for (size_t i = 0; i < n; i++)
	{
		xi = x[i];
		yi = y[i];
		ei = 1e-10;
		gsl_matrix_set (X, i, 0, 1.0);
		gsl_matrix_set (X, i, 1, xi);
		gsl_vector_set (yv, i, yi);
		gsl_vector_set (w, i, 1.0/(ei*ei));
	}

	gsl_multifit_linear_workspace * work = gsl_multifit_linear_alloc (n, 2);
	gsl_multifit_wlinear (X, w, yv, c, cov, &chisq, work);
	gsl_multifit_linear_free (work);
     
	outParams[0] = gsl_vector_get(c, 0);
	outParams[1] = gsl_vector_get(c, 1);
	printf ("# best fit: Y = %g + %g X\n", outParams[0], outParams[1]);

	if (gnuplotTitle != NULL)
	{
		double params3[3] = { outParams[0], outParams[1], 0 };
		DrawFitAndPlot(x, y, params3, gnuplotTitle);
	}
	
	gsl_matrix_free (X);
	gsl_vector_free (yv);
	gsl_vector_free (w);
	gsl_vector_free (c);
	gsl_matrix_free (cov);
}
