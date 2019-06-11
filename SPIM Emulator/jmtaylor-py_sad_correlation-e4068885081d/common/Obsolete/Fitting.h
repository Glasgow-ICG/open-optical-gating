/*
 *  Fitting.h
 *  fitting_expt
 *
 *  Created by Jonathan Taylor on 08/09/2011.
 *  Copyright 2011 Durham University. All rights reserved.
 *
 */

#include <vector>

void LinearFit(std::vector<double> x, std::vector<double> y, double outParams[2], const char *gnuplotTitle = NULL);
void QuadraticFit(std::vector<double> x, std::vector<double> y, double outParams[3], const char *gnuplotTitle = NULL);

