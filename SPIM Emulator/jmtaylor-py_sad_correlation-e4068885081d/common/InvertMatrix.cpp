/*	Module InvertMatrix.cpp

	Copyright 2010-2015 Jonathan Taylor. All rights reserved.

	Functions used to invert matrices (some of which make use of LAPACK)
	These are kept in a separate module because Accelerate.h conflicts with the GSL headers:
	this way there is more control about how things are compiled
 
	This is a bit of a mixed-up file, and it probably contains some overlap between functions 
	that do the same thing (for historical reasons!). I'm going to leave it as-is for now, though.
 */
	
#include "InvertMatrix.h"

#include "jOSMacros.h"
#include "jAssert.h"
#include "gsl/gsl_linalg.h"
#include "qr_complex.h"

#if OS_X
	#include <Accelerate/Accelerate.h>
#elif FEDORA_LINUX
	extern "C"
	{
		#include </opt/intel/mkl/10.0.1.014/include/mkl_lapack.h>
		typedef MKL_INT __CLPK_integer;
	}
#elif __athlon__
	#include <acml.h>
	void dgetrf_(int *m, int *n, double *a, int *lda, int *ipiv, int *info) { dgetrf(*m, *n, a, *lda, ipiv, info); }
	void dgetri_(int *n, double *a, int *lda, int *ipiv, int *info) { dgetri(*n, a, *lda, ipiv, info); }
	void sgetrf_(int *m, int *n, float *a, int *lda, int *ipiv, int *info) { sgetrf(*m, *n, a, *lda, ipiv, info); }
	void sgetri_(int *n, float *a, int *lda, int *ipiv, int *info) { sgetri(*n, a, *lda, ipiv, info); }
	void dtrtri_(char *uplo, char *diag, int *n, double *a, int *lda, int *info) { dtrtri(*uplo, *diag, *n, a, *lda, info); }
	void dtrtrs_(char *uplo, char *transa, char *diag, int *n, int *nrhs, double *a, int *lda, double *b, int *ldb, int *info) { dtrtrs(*uplo, *transa, *diag, *n, *nrhs, a, *lda, b, *ldb, info); }
	typedef int __CLPK_integer;
#elif 1
	// Until I come up with a reliable solution for providing a version of lapack, I'm just going to introduce stubs instead
	#define STUB 1
#else
	#include "clapack.h"
	typedef int __CLPK_integer;
#endif

#include <stdio.h>

#ifdef STUB

void Fail(void)
{
	printf("FATAL ERROR - not compiled against LAPACK so this matrix inversion function is not available\n");
	ALWAYS_ASSERT(0);
}

void InvertUpperTriangularMatrix(gsl_matrix *matrix) { Fail(); }
void SolveWithUpperTriangularMatrix(gsl_matrix *matrix, gsl_matrix *x) { Fail(); }
void InvertComplexMatrixUsingLAPACK(int inSize, jComplex *matrix) { Fail(); }

#else

void InvertComplexMatrixUsingLAPACK(int inSize, jComplex *matrix)
{
	__CLPK_integer *piv = new __CLPK_integer[inSize];
	__CLPK_integer info = 0;
	__CLPK_integer size = inSize, tda = inSize;
	

	zgetrf_(&size, &size, (__CLPK_doublecomplex*)matrix, &tda, piv, &info);
	ALWAYS_ASSERT(info == 0);

	__CLPK_integer workSize, minusOne = -1;
	jComplex zWorkSize;
	zgetri_(&size, (__CLPK_doublecomplex*)matrix, &tda, piv, (__CLPK_doublecomplex*)&zWorkSize, &minusOne, &info);
	ALWAYS_ASSERT(info == 0);
	workSize = (__CLPK_integer)zWorkSize.real();
	
	ALWAYS_ASSERT(workSize > 0);
	jComplex *work = new jComplex[workSize];
	zgetri_(&size, (__CLPK_doublecomplex*)matrix, &tda, piv, (__CLPK_doublecomplex*)work, &workSize, &info);
	ALWAYS_ASSERT(info == 0);
	delete[] work;
	delete[] piv;
}

void InvertMatrixUsingLAPACK(int inSize, double *matrix, int inTda)
{
	__CLPK_integer *piv = new __CLPK_integer[inSize];
	__CLPK_integer info = 0;
	__CLPK_integer size = inSize, tda = inTda;

	dgetrf_(&size, &size, matrix, &tda, piv, &info);
	ALWAYS_ASSERT(info == 0);

	__CLPK_integer workSize, minusOne = -1;
	double dWorkSize;
	dgetri_(&size, matrix, &tda, piv, &dWorkSize, &minusOne, &info);
	ALWAYS_ASSERT(info == 0);
	workSize = (__CLPK_integer)dWorkSize;

	ALWAYS_ASSERT(workSize > 0);
	double *work = new double[workSize];
	dgetri_(&size, matrix, &tda, piv, work, &workSize, &info);
	ALWAYS_ASSERT(info == 0);
	delete[] work;
	delete[] piv;
}

void InvertMatrixUsingLAPACK(int inSize, float *matrix, int inTda)
{
	__CLPK_integer *piv = new __CLPK_integer[inSize];
	__CLPK_integer info = 0;
	__CLPK_integer size = inSize, tda = inTda;

	sgetrf_(&size, &size, matrix, &tda, piv, &info);
	ALWAYS_ASSERT(info == 0);

	__CLPK_integer workSize, minusOne = -1;
	float sWorkSize;
	sgetri_(&size, matrix, &tda, piv, &sWorkSize, &minusOne, &info);
	ALWAYS_ASSERT(info == 0);
	workSize = (__CLPK_integer)sWorkSize;

	ALWAYS_ASSERT(workSize > 0);
	float *work = new float[workSize];
	sgetri_(&size, matrix, &tda, piv, work, &workSize, &info);
	ALWAYS_ASSERT(info == 0);
	delete[] work;
	delete[] piv;
}

void InvertUpperTriangularMatrixUsingLAPACK(int inSize, double *matrix, int inTda)
{
	__CLPK_integer info = 0;
	__CLPK_integer size = inSize, tda = inTda;

	// We are inverting an upper triangular matrix, but because LAPACK indexes in column-major order
	// (fortran style), we call it a lower triangular matrix here
	dtrtri_((char *)"L", (char *)"N", &size, matrix, &tda, &info);
	ALWAYS_ASSERT(info == 0);
}

void SolveWithUpperTriangularMatrixUsingLAPACK(int inSize1, int inSize2, double *matrix, int inTda, double *rhs, int inTdb)
{
	__CLPK_integer info = 0;
	__CLPK_integer size = inSize1, nRHS = inSize2, tda = inTda, tdb = inTdb;

	// We are solving with an upper triangular matrix, but because LAPACK indexes in column-major order
	// (fortran style), we call it a lower triangular matrix here
	dtrtrs_((char *)"L", (char *)"N", (char *)"N", &size, &nRHS, matrix, &tda, rhs, &tdb, &info);
	ALWAYS_ASSERT(info == 0);
}

void InvertUpperTriangularMatrix(gsl_matrix *matrix)
{
	InvertUpperTriangularMatrixUsingLAPACK(matrix->size1, matrix->data, matrix->tda);
}

void SolveWithUpperTriangularMatrix(gsl_matrix *matrix, gsl_matrix *x)
{
	SolveWithUpperTriangularMatrixUsingLAPACK(matrix->size1, x->size2, matrix->data, matrix->tda, x->data, x->tda);
}

#endif

