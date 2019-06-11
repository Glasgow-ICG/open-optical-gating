//
//  jCoord.cpp
//
//	Copyright 2010-2015 Jonathan Taylor. All rights reserved.
//
//	Implementations of utility functions for coordinate objects.
//	Most of the functions here are for coordinate transformations.
//
#include "jCoord.h"
#include "jUtils.h"

template<> void coord3T<double>::Print(const char *suffix) const
{
	printf("(%.12lg, %.12lg, %.12lg)%s", x, y, z, suffix);
}

#if JREAL_DEFINED
template<> void coord3T<jreal>::Print(const char *suffix) const
{
	printf("(");
	x.Print();
	printf(", ");
	y.Print();
	printf(", ");
	z.Print();
	printf(")%s", suffix);
}
#endif

void Print(coord3 c, const char *suffix)
{
	c.Print(suffix);
}

template<> void coordC3T<jComplex, double>::Print(const char *suffix) const
{
	printf("({%.12lg,%.12lg}, {%.12lg,%.12lg}, {%.12lg,%.12lg})%s", x.real(), x.imag(), y.real(), y.imag(), z.real(), z.imag(), suffix);
}

#if JREAL_DEFINED
template<> void coordC3T<jComplexR, jreal>::Print(const char *suffix) const
{
	printf("(");
	::Print(x);
	printf(", ");
	::Print(y);
	printf(", ");
	::Print(z);
	printf(")%s", suffix);
}
#endif

void Print(coordC3 c, const char *suffix)
{
	c.Print(suffix);
}

coordC3 RotateFromSphericalSystem(coordC3 c, double theta, double phi)
{
	c.RotateFromSphericalSystem(theta, phi);
	return c;	
}

coordC3 RotateToSphericalSystem(coordC3 c, double theta, double phi)
{
	c.RotateToSphericalSystem(theta, phi);
	return c;	
}

coord3 RotateFromSphericalSystem(coord3 c, double theta, double phi)
{
	c.RotateFromSphericalSystem(theta, phi);
	return c;	
}

coord3 RotateToSphericalSystem(coord3 c, double theta, double phi)
{
	c.RotateToSphericalSystem(theta, phi);
	return c;	
}

coordC3 RotateFromCylindricalSystem(coordC3 c, double phi)
{
	c.RotateFromCylindricalSystem(phi);
	return c;	
}

coordC3 RotateToCylindricalSystem(coordC3 c, double phi)
{
	c.RotateToCylindricalSystem(phi);
	return c;	
}

template<> coord3 CartesianToSpherical(coord3 source)
{
	// Convert from a Cartesian coordinate to a spherical coordinate
	double	r = sqrt(source.x*source.x + source.y*source.y + source.z*source.z);
	double	theta = acos(source.z / r);
	double	phi = atan2(source.y, source.x);
	
	return coord3(r, theta, phi);
}

template<> coord3 SphericalToCartesian(coord3 source)
{
	// Convert from a spherical coordinate to a Cartesian coordinate
	double	cartX = source.x * cos(source.z) * sin(source.y);
	double	cartY = source.x * sin(source.z) * sin(source.y);
	double	cartZ = source.x * cos(source.y);
	
	return coord3(cartX, cartY, cartZ);
}

#ifdef USE_JREAL
template<> coord3R CartesianToSpherical(coord3R source)
{
	// Convert from a Cartesian coordinate to a spherical coordinate
	jreal	r = sqrt(source.x*source.x + source.y*source.y + source.z*source.z);
	jreal	theta = acos(source.z / r);
	jreal	phi = atan2(source.y, source.x);
	
	return coord3R(r, theta, phi);
}

template<> coord3R SphericalToCartesian(coord3R source)
{
	// Convert from a spherical coordinate to a Cartesian coordinate
	jreal	cartX = source.x * cos(source.z) * sin(source.y);
	jreal	cartY = source.x * sin(source.z) * sin(source.y);
	jreal	cartZ = source.x * cos(source.y);
	
	return coord3R(cartX, cartY, cartZ);
}
#endif

#if 0
/*	Disabled as this was just a work in progress
	As per the note below, I am not sure what it even means to have a complex spherical vector!	*/
coordC3 CartesianToSpherical(coordC3 source)
{
	// NOTE: To be honest I'm not even sure what it means to have a complex spherical vector,
	// as well as specific worries such as whether r should use x^2 or |x|^2, for example
	// This might have been of interest with evanescent waves, but I think I'll have to leave it for now!
	// As for the r part, I'm pretty sure I shouldn't use |x|^2...
	jComplex r = sqrt(source.x*source.x + source.y*source.y + source.z*source.z);
	jComplex theta = cacos(source.z / r);
	jComplex val = source.y / source.x;		// **** not sure what to do about quadrants (c.f. atan2...)
	jComplex complexAtan = -jComplex::i() * 0.5 * log((1.0 + jComplex::i() * val) / (1.0 - jComplex::i() * val));
	jComplex phi = complexAtan;
	
	return coordC3(r, theta, phi);
}

coordC3 SphericalToCartesian(coordC3 source)
{
	jComplex cartX = source.x * cos(source.z) * sin(source.y);
	jComplex cartY = source.x * sin(source.z) * sin(source.y);
	jComplex cartZ = source.x * cos(source.y);
	
	return coordC3(cartX, cartY, cartZ);
}
#endif

coord3 CartesianToCylindrical(coord3 source)
{
	double	r = sqrt(source.x*source.x + source.y*source.y);
	double	theta = atan2(source.y, source.x);
	
	return coord3(r, theta, source.z);
}

coord3 CylindricalToCartesian(coord3 source)
{
	double	cartX = source.x * cos(source.y);
	double	cartY = source.x * sin(source.y);
	double	cartZ = source.z;
	
	return coord3(cartX, cartY, cartZ);
}

coord2 coord2::RotateByRadians(double radians)
{
    double newX = x * cos(radians) - y * sin(radians);
    double newY = x * sin(radians) + y * cos(radians);
    return coord2(newX, newY);
}

#ifdef __GSL_COMPLEX_H__
template<> gsl_vector *coord3T<double>::AllocGSLVector(void) const
{
	gsl_vector	*resultVector = gsl_vector_calloc(3);
	gsl_vector_set(resultVector, 0, x);
	gsl_vector_set(resultVector, 1, y);
	gsl_vector_set(resultVector, 2, z);
	return resultVector;
}

template<> coord3T<double>::coord3T(gsl_vector *inVector)
{
	x = gsl_vector_get(inVector, 0);
	y = gsl_vector_get(inVector, 1);
	z = gsl_vector_get(inVector, 2);
}

template<> coord3T<double>::coord3T(gsl_vector *inVector, int offset)
{
	x = gsl_vector_get(inVector, offset);
	y = gsl_vector_get(inVector, offset+1);
	z = gsl_vector_get(inVector, offset+2);
}

#endif

