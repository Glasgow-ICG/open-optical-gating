// JMT: modified the GSL functions for QR decomposition to work for complex variables


/* linalg/qr.c
 * 
 * Copyright (C) 1996, 1997, 1998, 1999, 2000 Gerard Jungman, Brian Gough
 * 
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or (at
 * your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
 */

/* Author:  G. Jungman */


//#include <config.h>
#include <stdlib.h>
#include <string.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>

#include <gsl/gsl_linalg.h>

#include "qr_complex.h"

/* Factorise a general M x N matrix A into
 *  
 *   A = Q R
 *
 * where Q is orthogonal (M x M) and R is upper triangular (M x N).
 *
 * Q is stored as a packed set of Householder transformations in the
 * strict lower triangular part of the input matrix.
 *
 * R is stored in the diagonal and upper triangle of the input matrix.
 *
 * The full matrix for Q can be obtained as the product
 *
 *       Q = Q_k .. Q_2 Q_1
 *
 * where k = MIN(M,N) and
 *
 *       Q_i = (I - tau_i * v_i * v_i')
 *
 * and where v_i is a Householder vector
 *
 *       v_i = [1, m(i+1,i), m(i+2,i), ... , m(M,i)]
 *
 * This storage scheme is the same as in LAPACK.  */

int
gsl_linalg_complex_QR_decomp (gsl_matrix_complex * A, gsl_vector_complex * tau)
{
  const size_t M = A->size1;
  const size_t N = A->size2;

  if (tau->size != GSL_MIN (M, N))
    {
      GSL_ERROR ("size of tau must be MIN(M,N)", GSL_EBADLEN);
    }
  else
    {
      size_t i;

      for (i = 0; i < GSL_MIN (M, N); i++)
        {
          /* Compute the Householder transformation to reduce the j-th
             column of the matrix to a multiple of the j-th unit vector */

          gsl_vector_complex_view c_full = gsl_matrix_complex_column (A, i);
          gsl_vector_complex_view c = gsl_vector_complex_subvector (&(c_full.vector), i, M-i);

          gsl_complex tau_i = gsl_complex_conjugate(gsl_linalg_complex_householder_transform (&(c.vector)));

          gsl_vector_complex_set (tau, i, tau_i);

          /* Apply the transformation to the remaining columns and
             update the norms */

          if (i + 1 < N)
            {
              gsl_matrix_complex_view m = gsl_matrix_complex_submatrix (A, i, i + 1, M - i, N - (i + 1));
              gsl_linalg_complex_householder_hm (tau_i, &(c.vector), &(m.matrix));
            }
        }

      return GSL_SUCCESS;
    }
}

/* Solves the system A x = b in place using the QR factorisation,

 *  R x = Q^T b
 *
 * to obtain x. Based on SLATEC code. 
 */

int
gsl_linalg_complex_QR_svx (const gsl_matrix_complex * QR, const gsl_vector_complex * tau, gsl_vector_complex * x)
{

  if (QR->size1 != QR->size2)
    {
      GSL_ERROR ("QR matrix must be square", GSL_ENOTSQR);
    }
  else if (QR->size1 != x->size)
    {
      GSL_ERROR ("matrix size must match x/rhs size", GSL_EBADLEN);
    }
  else
    {
      /* compute rhs = Q^T b */

      gsl_linalg_complex_QR_QTvec (QR, tau, x);

      /* Solve R x = rhs, storing x in-place */

      gsl_blas_ztrsv (CblasUpper, CblasNoTrans, CblasNonUnit, QR, x);

      return GSL_SUCCESS;
    }
}


/* Find the least squares solution to the overdetermined system 
 *
 *   A x = b 
 *  
 * for M >= N using the QR factorization A = Q R. 
 */

int
gsl_linalg_complex_QR_lssolve (const gsl_matrix_complex * QR, const gsl_vector_complex * tau, const gsl_vector_complex * b, gsl_vector_complex * x, gsl_vector_complex * residual)
{
  const size_t M = QR->size1;
  const size_t N = QR->size2;

  if (M < N)
    {
      GSL_ERROR ("QR matrix must have M>=N", GSL_EBADLEN);
    }
  else if (M != b->size)
    {
      GSL_ERROR ("matrix size must match b size", GSL_EBADLEN);
    }
  else if (N != x->size)
    {
      GSL_ERROR ("matrix size must match solution size", GSL_EBADLEN);
    }
  else if (M != residual->size)
    {
      GSL_ERROR ("matrix size must match residual size", GSL_EBADLEN);
    }
  else
    {
      gsl_matrix_complex_const_view R = gsl_matrix_complex_const_submatrix (QR, 0, 0, N, N);
      gsl_vector_complex_view c = gsl_vector_complex_subvector(residual, 0, N);

      gsl_vector_complex_memcpy(residual, b);

      /* compute rhs = Q^T b */

      gsl_linalg_complex_QR_QTvec (QR, tau, residual);

      /* Solve R x = rhs */

      gsl_vector_complex_memcpy(x, &(c.vector));

      gsl_blas_ztrsv (CblasUpper, CblasNoTrans, CblasNonUnit, &(R.matrix), x);

      /* Compute residual = b - A x = Q (Q^T b - R x) */
      
      gsl_vector_complex_set_zero(&(c.vector));

      gsl_linalg_complex_QR_Qvec(QR, tau, residual);

      return GSL_SUCCESS;
    }
}

/* Form the product Q^T v  from a QR factorized matrix 
 */

int
gsl_linalg_complex_QR_QTvec (const gsl_matrix_complex * QR, const gsl_vector_complex * tau, gsl_vector_complex * v)
{
  const size_t M = QR->size1;
  const size_t N = QR->size2;

  if (tau->size != GSL_MIN (M, N))
    {
      GSL_ERROR ("size of tau must be MIN(M,N)", GSL_EBADLEN);
    }
  else if (v->size != M)
    {
      GSL_ERROR ("vector size must be N", GSL_EBADLEN);
    }
  else
    {
      size_t i;

      /* compute Q^T v */

      for (i = 0; i < GSL_MIN (M, N); i++)
        {
          gsl_vector_complex_const_view c = gsl_matrix_complex_const_column (QR, i);
          gsl_vector_complex_const_view h = gsl_vector_complex_const_subvector (&(c.vector), i, M - i);
          gsl_vector_complex_view w = gsl_vector_complex_subvector (v, i, M - i);
          gsl_complex ti = gsl_vector_complex_get (tau, i);
//		  printf("%ld %ld %ld %ld\n", M, N, i, h.vector.size);
//		  if (h.vector.size == 1)
//			Debugger();
          gsl_linalg_complex_householder_hv (ti, &(h.vector), &(w.vector));
        }
      return GSL_SUCCESS;
    }
}


int
gsl_linalg_complex_QR_Qvec (const gsl_matrix_complex * QR, const gsl_vector_complex * tau, gsl_vector_complex * v)
{
  const size_t M = QR->size1;
  const size_t N = QR->size2;

  if (tau->size != GSL_MIN (M, N))
    {
      GSL_ERROR ("size of tau must be MIN(M,N)", GSL_EBADLEN);
    }
  else if (v->size != M)
    {
      GSL_ERROR ("vector size must be N", GSL_EBADLEN);
    }
  else
    {
      size_t i;

      /* compute Q^T v */

      for (i = GSL_MIN (M, N); i > 0 && i--;)
        {
          gsl_vector_complex_const_view c = gsl_matrix_complex_const_column (QR, i);
          gsl_vector_complex_const_view h = gsl_vector_complex_const_subvector (&(c.vector), 
                                                                i, M - i);
          gsl_vector_complex_view w = gsl_vector_complex_subvector (v, i, M - i);
          gsl_complex ti = gsl_vector_complex_get (tau, i);
          gsl_linalg_complex_householder_hv (ti, &h.vector, &w.vector);
        }
      return GSL_SUCCESS;
    }
}


/*  Form the orthogonal matrix Q from the packed QR matrix */

int
gsl_linalg_complex_QR_unpack (const gsl_matrix_complex * QR, const gsl_vector_complex * tau, gsl_matrix_complex * Q, gsl_matrix_complex * R)
{
  const size_t M = QR->size1;
  const size_t N = QR->size2;

  if (Q->size1 != M || Q->size2 != M)
    {
      GSL_ERROR ("Q matrix must be M x M", GSL_ENOTSQR);
    }
  else if (R->size1 != M || R->size2 != N)
    {
      GSL_ERROR ("R matrix must be M x N", GSL_ENOTSQR);
    }
  else if (tau->size != GSL_MIN (M, N))
    {
      GSL_ERROR ("size of tau must be MIN(M,N)", GSL_EBADLEN);
    }
  else
    {
      size_t i, j;

      /* Initialize Q to the identity */

      gsl_matrix_complex_set_identity (Q);

      for (i = GSL_MIN (M, N); i > 0 && i--;)
        {
          gsl_vector_complex_const_view c = gsl_matrix_complex_const_column (QR, i);
          gsl_vector_complex_const_view h = gsl_vector_complex_const_subvector (&c.vector,
                                                                i, M - i);
          gsl_matrix_complex_view m = gsl_matrix_complex_submatrix (Q, i, i, M - i, M - i);
          gsl_complex ti = gsl_vector_complex_get (tau, i);
          gsl_linalg_complex_householder_hm (gsl_complex_conjugate(ti), &h.vector, &m.matrix);
        }

      /*  Form the right triangular matrix R from a packed QR matrix */

      for (i = 0; i < M; i++)
        {
          for (j = 0; j < i && j < N; j++)
            gsl_matrix_complex_set (R, i, j, gsl_complex_rect(0.0, 0.0));

          for (j = i; j < N; j++)
            gsl_matrix_complex_set (R, i, j, gsl_matrix_complex_get (QR, i, j));
        }

      return GSL_SUCCESS;
    }
}
