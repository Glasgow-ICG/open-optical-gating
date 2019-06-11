/*
 *  InvertMatrix.h
 *
 *	Copyright 2010-2015 Jonathan Taylor. All rights reserved.
 *
 */


void InvertUpperTriangularMatrix(gsl_matrix *matrix);
void SolveWithUpperTriangularMatrix(gsl_matrix *matrix, gsl_matrix *x);
void InvertComplexMatrixUsingLAPACK(int inSize, jComplex *matrix);

#ifdef __GSL_MATRIX_H__
	void InvertMatrix(gsl_matrix *matrix, int size);
	void InvertMatrix(gsl_matrix_complex *matrix, int size);
	jComplex *qr_solve_complex_with_real(gsl_matrix_complex *m, gsl_vector_complex *a);
	jComplex *qr_solve_complex(gsl_matrix_complex *m, gsl_vector_complex *a, bool useGMRES = false);
#endif
