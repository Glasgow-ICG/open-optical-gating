#ifndef __QR_COMPLEX_H__
#define __QR_COMPLEX_H__

int gsl_linalg_complex_QR_decomp (gsl_matrix_complex * A, gsl_vector_complex * tau);
int gsl_linalg_complex_QR_lssolve (const gsl_matrix_complex * QR, const gsl_vector_complex * tau, const gsl_vector_complex * b, gsl_vector_complex * x, gsl_vector_complex * residual);
int gsl_linalg_complex_QR_svx (const gsl_matrix_complex * QR, const gsl_vector_complex * tau, gsl_vector_complex * x);
int gsl_linalg_complex_QR_QTvec (const gsl_matrix_complex * QR, const gsl_vector_complex * tau, gsl_vector_complex * v);
int gsl_linalg_complex_QR_Qvec (const gsl_matrix_complex * QR, const gsl_vector_complex * tau, gsl_vector_complex * v);
int gsl_linalg_complex_QR_unpack (const gsl_matrix_complex * QR, const gsl_vector_complex * tau, gsl_matrix_complex * Q, gsl_matrix_complex * R);

#endif
