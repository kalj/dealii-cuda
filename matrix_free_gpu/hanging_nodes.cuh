/* -*- c-basic-offset:2; tab-width:2; indent-tabs-mode:nil -*-
 *
 * @(#)hanging_nodes.cuh
 * @author Karl Ljungkvist <karl.ljungkvist@it.uu.se>
 *
 */

#ifndef __deal2__matrix_free_hanging_nodes_h
#define __deal2__matrix_free_hanging_nodes_h

#include <deal.II/base/types.h>
#include <deal.II/dofs/dof_handler.h>
#include "defs.h"
#include "utils.h"
#include "cuda_utils.cuh"

__constant__ double constraint_weights[(MAX_ELEM_DEGREE+1)*(MAX_ELEM_DEGREE+1)];

using namespace dealii;
// element has a constraint at *==0 or *==p face
#define CONSTR_X      (1<<0)
#define CONSTR_Y      (1<<1)
#define CONSTR_Z      (1<<2)

// The element is placed in the 'first position' along *-axis.
// these also determine which face is constrained. For example, in 2D, if
// CONSTR_X and CONSTR_X_TYPE is set, then x==0 is constrained.
#define CONSTR_X_TYPE (1<<3)
#define CONSTR_Y_TYPE (1<<4)
#define CONSTR_Z_TYPE (1<<5)

#define NOTRANSPOSE false
#define TRANSPOSE true

//=============================================================================
// setup functions
//=============================================================================

template <int dim>
void get_lex_face_mapping(std::vector<types::global_dof_index> &mapping, const unsigned int p)
{

  const unsigned int n_dofs_1d = p+1;
  if(dim==2)
  {
    // setup mapping to be 0,2,3,...,1
    mapping.resize(n_dofs_1d);
    mapping[0] = 0;
    mapping[p] = 1;

    for(int i = 0; i <p-1 ; ++i) {
      mapping[i+1] = i+2;
    }

  }
  else if(dim==3)
  {
    // setup mapping to be
    // e.g. p=3
    // 2 10 11 3
    // 5 14 15 7
    // 4 12 13 6
    // 0  8  9 1

    mapping.resize(n_dofs_1d*n_dofs_1d);
    mapping[0] = 0;
    mapping[p] = 1;
    mapping[p*n_dofs_1d] = 2;
    mapping[n_dofs_1d*n_dofs_1d-1] = 3;

    int k=4;
    for(int i = 1; i <p ; ++i, k++)
      mapping[i*n_dofs_1d] = k;

    for(int i = 1; i <p ; ++i, k++)
      mapping[i*n_dofs_1d+p] = k;

    for(int i = 1; i <p ; ++i, k++)
      mapping[i] = k;

    for(int i = 1; i <p ; ++i, k++)
      mapping[i+p*n_dofs_1d] = k;

    for(int i = 1; i <p ; ++i)
      for(int j = 1; j <p ; ++j, k++)
        mapping[i*n_dofs_1d+j] = k;

  }
  else {
    ExcNotImplemented();
  }
}

void setup_constraint_weights(unsigned int fe_degree) {
  FE_Q<2> fe_q(fe_degree);
  FullMatrix<double> interpolation_matrix(fe_q.dofs_per_face,
                                          fe_q.dofs_per_face);

  fe_q.get_subface_interpolation_matrix(fe_q,0,interpolation_matrix);

  std::vector<unsigned int> mapping(fe_degree+1);
  get_lex_face_mapping<2>(mapping,fe_degree);

  FullMatrix<double> mapped_matrix(fe_q.dofs_per_face,
                                   fe_q.dofs_per_face);

  mapped_matrix.fill_permutation(interpolation_matrix,mapping,mapping);

  CUDA_CHECK_SUCCESS(cudaMemcpyToSymbol(constraint_weights, &mapped_matrix[0][0],
                                        sizeof(double)*fe_q.dofs_per_face*fe_q.dofs_per_face));
}


void rotate_subface_index(unsigned int &subface_index, int times)
{
  const unsigned int rot_mapping[4] = {2, 0, 3, 1};

  times = times % 4;
  times = times < 0 ? times + 4 : times;
  for(int t = 0; t < times; ++t) {
    subface_index = rot_mapping[subface_index];
  }
}


void rotate_face(std::vector<types::global_dof_index> &dofs, int times, unsigned int n_dofs_1d)
{
  const unsigned int rot_mapping[4] = {2, 0, 3, 1};

  times = times % 4;
  times = times < 0 ? times + 4 : times;

  std::vector<types::global_dof_index> copy(dofs.size());

  for(int t = 0; t < times; ++t) {

    std::swap(copy,dofs);

    // vertices
    for(int i = 0; i < 4; ++i)
      dofs[rot_mapping[i]] = copy[i];

    // edges
    const unsigned int n_int = n_dofs_1d-2;
    unsigned int offset = 4;

    // west edge
    for(int i = 0; i < n_int; ++i)
      dofs[offset + 0*n_int + i] = copy[offset + 2*n_int + (n_int-1-i)];
    // east edge
    for(int i = 0; i < n_int; ++i)
      dofs[offset + 1*n_int + i] = copy[offset + 3*n_int + (n_int-1-i)];
    // south edge
    for(int i = 0; i < n_int; ++i)
      dofs[offset + 2*n_int + i] = copy[offset + 1*n_int + i];
    // north edge
    for(int i = 0; i < n_int; ++i)
      dofs[offset + 3*n_int + i] = copy[offset + 0*n_int + i];

    // interior points
    offset += 4*n_int;

    for(int i = 0; i <n_int ; ++i)
      for(int j = 0; j <n_int ; ++j)
        // dofs[offset + i*n_int + j] = copy[offset + (n_int-1-j)*n_int + i];
        dofs[offset + i*n_int + j] = copy[offset + j*n_int + (n_int-1-i)];

  }
}

void transpose_face(std::vector<types::global_dof_index> &dofs, unsigned int n_dofs_1d)
{
  const std::vector<types::global_dof_index> copy(dofs);

  // vertices
  // dofs[0] = copy[0];
  dofs[1] = copy[2];
  dofs[2] = copy[1];
  // dofs[3] = copy[3];

  // edges
  const unsigned int n_int = n_dofs_1d-2;
  unsigned int offset = 4;

  // west edge
  for(int i = 0; i < n_int; ++i)
    dofs[offset + 0*n_int + i] = copy[offset + 2*n_int + i];
  // east edge
  for(int i = 0; i < n_int; ++i)
    dofs[offset + 1*n_int + i] = copy[offset + 3*n_int + i];
  // south edge
  for(int i = 0; i < n_int; ++i)
    dofs[offset + 2*n_int + i] = copy[offset + 0*n_int + i];
  // north edge
  for(int i = 0; i < n_int; ++i)
    dofs[offset + 3*n_int + i] = copy[offset + 1*n_int + i];

  // interior points
  offset += 4*n_int;

  for(int i = 0; i <n_int ; ++i)
    for(int j = 0; j <n_int ; ++j)
      dofs[offset + i*n_int + j] = copy[offset + j*n_int + i];

}

void transpose_subface_index(unsigned int &subface)
{
  if(subface == 1) subface = 2;
  else if(subface == 2) subface = 1;
}


template <int dim, typename T>
void setup_hanging_node_constraints(unsigned int &mask,
                                    std::vector<unsigned int> &dof_indices,
                                    const unsigned int p,
                                    const T &cell, unsigned int cellid)
{
  mask = 0;
  const unsigned int n_dofs_1d = p+1;
  const unsigned int dofs_per_face = ipowf(n_dofs_1d,dim-1);

  std::vector<types::global_dof_index> neighbor_dofs(dofs_per_face);

  std::vector<unsigned int> lex_face_mapping;
  get_lex_face_mapping<dim> (lex_face_mapping,p);

  for (unsigned int face=0; face<GeometryInfo<dim>::faces_per_cell; ++face) {
    const typename DoFHandler<dim>::active_cell_iterator &
      neighbor = cell->neighbor(face);

      // neighbor is coarser than we, i.e. face f is constrained!
    if (!cell->at_boundary(face) &&
        neighbor->level() < cell->level())
    {
      const unsigned int neighbor_face = cell->neighbor_face_no(face);

      // find position of face on neighbor
      unsigned int subface=0;
      for (; subface< GeometryInfo<dim>::max_children_per_face;
           ++subface)
        if(neighbor->neighbor_child_on_subface(neighbor_face,subface) == cell)
          break;

      // get indices to read
      neighbor->face(neighbor_face)->get_dof_indices(neighbor_dofs);

      if(dim==2) {

        if(face < 2) {
          mask |= CONSTR_X;
          if(face==0)
            mask |= CONSTR_X_TYPE;
          if(subface==0)
            mask |= CONSTR_Y_TYPE;
        }
        else {
          mask |= CONSTR_Y;
          if(face==2)
            mask |= CONSTR_Y_TYPE;
          if(subface==0)
            mask |= CONSTR_X_TYPE;
        }

        // reorder neighbor_dofs and copy into face'th face of dof_indices

        // offset if upper/right face
        unsigned int offset = (face%2==1) ? p : 0;

        for(int i = 0; i < n_dofs_1d; ++i) {
          unsigned int idx;
          // if X-line, i.e. if y==0 or y==p, need to copy like this:
          if(face>1)
            idx = n_dofs_1d*offset + i;
          // if Y-line, i.e. if x==0 or x==p, need to copy like this:
          else
            idx = n_dofs_1d*i + offset;

          dof_indices[idx] = neighbor_dofs[lex_face_mapping[i]];
        }

      }
      else if(dim==3) {

        // we know the indices, and what face is constrained, but we have to be
        // careful with the type, and the orientation of the indices


        // if it's flipped, we have to transpose the face
        bool transpose = !(cell->face_orientation(face));

        // possibly rotate:
        int rotate = 0;

        if(cell->face_rotation(face))
          rotate -= 1;
        if(cell->face_flip(face))
          rotate -= 2;



        rotate_face(neighbor_dofs,rotate,n_dofs_1d);
        rotate_subface_index(subface,rotate);

        if(transpose) transpose_face(neighbor_dofs,n_dofs_1d);
        if(transpose) transpose_subface_index(subface);



        // YZ-plane
        if(face < 2) {
          mask |= CONSTR_X;
          if(face==0)
            mask |= CONSTR_X_TYPE;
          if(subface%2==0)
            mask |= CONSTR_Y_TYPE;
          if(subface/2==0)
            mask |= CONSTR_Z_TYPE;
        }
        // XZ-plane
        else if(face < 4) {
          mask |= CONSTR_Y;
          if(face==2)
            mask |= CONSTR_Y_TYPE;
          if(subface%2==0)
            mask |= CONSTR_Z_TYPE;
          if(subface/2==0)
            mask |= CONSTR_X_TYPE;
        }
        // XY-plane
        else {
          mask |= CONSTR_Z;
          if(face==4)
            mask |= CONSTR_Z_TYPE;
          if(subface%2==0)
            mask |= CONSTR_X_TYPE;
          if(subface/2==0)
            mask |= CONSTR_Y_TYPE;
        }

        // offset if upper/right/back face
        unsigned int offset = (face%2==1) ? p : 0;

        for(int i = 0; i < n_dofs_1d; ++i) {
          for(int j = 0; j < n_dofs_1d; ++j) {
            unsigned int idx;
            // if YZ-plane, i.e. if x==0 or x==p, and orientation standard, need to copy like this:
            if(face<2)
              idx = n_dofs_1d*n_dofs_1d*i + n_dofs_1d*j + offset;
            // if XZ-plane, i.e. if y==0 or y==p, and orientation standard, need to copy like this:
            else if(face<4)
              idx = n_dofs_1d*n_dofs_1d*j + n_dofs_1d*offset + i;
            // if XY-plane, i.e. if z==0 or z==p, and orientation standard, need to copy like this:
            else
              idx = n_dofs_1d*n_dofs_1d*offset + n_dofs_1d*i + j;

            dof_indices[idx] = neighbor_dofs[lex_face_mapping[n_dofs_1d*i+j]];
          }
        }

      }
      else {
        ExcNotImplemented();
      }


    }
  }
}


//=============================================================================
// Functions for resolving the hanging node constraints
//=============================================================================

template <unsigned int size>
__device__ inline unsigned int index2(unsigned int i, unsigned int j) {
  return i+size*j;
}

template <unsigned int size>
__device__ inline unsigned int index3(unsigned int i, unsigned int j, unsigned int k) {
  return i+size*j+size*size*k;
}


template<unsigned int fe_degree, unsigned int direction, bool transpose, typename Number>
__device__ inline void interpolate_boundary_3d(Number *values, const unsigned int constr)
{
  const unsigned int xidx = threadIdx.x % (fe_degree+1);
  const unsigned int yidx = threadIdx.y;
  const unsigned int zidx = threadIdx.z;

  const unsigned int CONSTR_TYPE_MASK = (direction==0) ? CONSTR_X_TYPE : (direction==1) ? CONSTR_Y_TYPE : CONSTR_Z_TYPE;

  // if computing in x-direction, need to match against CONSTR_Y or CONSTR_Z
  if( constr &
      (((direction==0) ? 0 : CONSTR_X) |
       ((direction==1) ? 0 : CONSTR_Y) |
       ((direction==2) ? 0 : CONSTR_Z))
      ) {

    const unsigned int interp_idx = (direction==0) ? xidx : (direction==1) ? yidx : zidx;

    syncthreads();

    Number t = 0;

    const bool flag =
      ((direction != 0) && ((constr & CONSTR_X) && ((constr & CONSTR_X_TYPE) ? (xidx==0) : (xidx==fe_degree)))) ||
      ((direction != 1) && ((constr & CONSTR_Y) && ((constr & CONSTR_Y_TYPE) ? (yidx==0) : (yidx==fe_degree)))) ||
      ((direction != 2) && ((constr & CONSTR_Z) && ((constr & CONSTR_Z_TYPE) ? (zidx==0) : (zidx==fe_degree))));

    if(flag) {
      const bool type = constr & CONSTR_TYPE_MASK;

      if(type) {
        for(int i=0; i<=fe_degree; i++) {// read as usual

          const unsigned int realidx =
            (direction==0) ? index3<fe_degree+1>(i,yidx,zidx) :
            (direction==1) ? index3<fe_degree+1>(xidx,i,zidx) :
            index3<fe_degree+1>(xidx,yidx,i);

          const Number w =
            transpose ? constraint_weights[i*(fe_degree+1) + interp_idx] :
            constraint_weights[interp_idx*(fe_degree+1) + i];
          t += w*values[realidx];
        }
      }
      else {
        for(int i=0; i<=fe_degree; i++) {

          const unsigned int realidx =
            (direction==0) ? index3<fe_degree+1>(i,yidx,zidx) :
            (direction==1) ? index3<fe_degree+1>(xidx,i,zidx) :
            index3<fe_degree+1>(xidx,yidx,i);

          const Number w =
            transpose ? constraint_weights[(fe_degree-i)*(fe_degree+1) + fe_degree-interp_idx] :
            constraint_weights[(fe_degree-interp_idx)*(fe_degree+1) + fe_degree-i];
          t += w*values[realidx];
        }
      }
    }

    syncthreads();

    if(flag)
      values[index3<fe_degree+1>(xidx,yidx,zidx)] = t;
  }
}

template<unsigned int fe_degree, unsigned int direction, bool transpose, typename Number>
__device__ inline void interpolate_boundary_2d(Number *values, const unsigned int constr)
{
  const unsigned int xidx = threadIdx.x % (fe_degree+1);
  const unsigned int yidx = threadIdx.y;

  const unsigned int CONSTR_TYPE_MASK = (direction==0) ? CONSTR_X_TYPE : CONSTR_Y_TYPE;

  if(constr &
     (((direction==0) ? CONSTR_Y : 0) |
      ((direction==1) ? CONSTR_X : 0))
     ) {

    const unsigned int interp_idx = (direction==0) ? xidx : yidx;

    syncthreads();

    Number t = 0;
    const bool flag =
      ((direction==0) && ((constr & CONSTR_Y_TYPE) ? (yidx==0) : (yidx==fe_degree))) ||
      ((direction==1) && ((constr & CONSTR_X_TYPE) ? (xidx==0) : (xidx==fe_degree)));

    if(flag) {
      const bool type = constr & CONSTR_TYPE_MASK;

      if(type)
        for(int i=0; i<=fe_degree; i++) {// read as usual
          const unsigned int realidx =
            (direction==0) ? index2<fe_degree+1>(i,yidx) :
            index2<fe_degree+1>(xidx,i);

          const Number w =
            transpose ? constraint_weights[i*(fe_degree+1) + interp_idx] :
            constraint_weights[interp_idx*(fe_degree+1) + i];
          t += w*values[realidx];
        }
      else
        for(int i=0; i<=fe_degree; i++) { // read reversed
          const unsigned int realidx =
            (direction==0) ? index2<fe_degree+1>(i,yidx) :
            index2<fe_degree+1>(xidx,i);

          const Number w =
            transpose ? constraint_weights[(fe_degree-i)*(fe_degree+1) + fe_degree-interp_idx] :
            constraint_weights[(fe_degree-interp_idx)*(fe_degree+1) + fe_degree-i];
          t += w*values[realidx];
        }
    }

    syncthreads();

    if(flag) values[index2<fe_degree+1>(xidx,yidx)] = t;
  }
}

template <int dim, int fe_degree, bool transpose, typename Number>
__device__ void resolve_hanging_nodes_shmem(Number *values, const unsigned int constr)
{
  if(dim==2) {
    interpolate_boundary_2d<fe_degree,0,transpose>(values,constr);
    interpolate_boundary_2d<fe_degree,1,transpose>(values,constr);
  }
  else if(dim==3) {
    // first along x-direction
    // interpolate y and z faces
    interpolate_boundary_3d<fe_degree,0,transpose>(values,constr);
    // now along y-direction
    // interpolate x and z faces
    interpolate_boundary_3d<fe_degree,1,transpose>(values,constr);
    // now along z-direction
    // interpolate x and y faces
    interpolate_boundary_3d<fe_degree,2,transpose>(values,constr);
  }
}


template<unsigned int fe_degree, unsigned int direction, bool transpose, typename Number>
__device__ inline void interpolate_boundary_3d_pmem(Number *values, const unsigned int constr)
{
  const unsigned int n = fe_degree+1;

  const unsigned int THIS_TYPE = direction==0 ? CONSTR_X_TYPE :
    (direction==1 ? CONSTR_Y_TYPE : CONSTR_Z_TYPE);

  const unsigned int FIRST_CONSTR = direction==0 ? CONSTR_Z :
    (direction==1 ? CONSTR_X : CONSTR_X);
  const unsigned int SECOND_CONSTR = direction==0 ? CONSTR_Y :
    (direction==1 ? CONSTR_Z : CONSTR_Y);

  const unsigned int FIRST_TYPE = direction==0 ? CONSTR_Z_TYPE :
    (direction==1 ? CONSTR_X_TYPE : CONSTR_X_TYPE );
  const unsigned int SECOND_TYPE = direction==0 ? CONSTR_Y_TYPE :
    (direction==1 ? CONSTR_Z_TYPE : CONSTR_Y_TYPE );

  Number facevals[n];

  if(constr & FIRST_CONSTR) { // xy-planes

    const unsigned int offset = (constr & FIRST_TYPE) ? 0 : fe_degree;
    const bool type = constr & THIS_TYPE; // if ==1 face is of 'first' type

    for(int i = 0; i < n; ++i) { // loop in y-direction
      for(int j = 0; j < n; ++j) { // loop in x-direction

        facevals[j] = 0;
        for(int k = 0; k < n; ++k) {
          const unsigned int readidx = direction==0 ? index3<n>(k,i,offset) :
            (direction==1 ? index3<n>(offset,k,i) : index3<n>(offset,i,k));

          // if type==0, read reversed
          const unsigned int idx1 = transpose ? j : k;
          const unsigned int idx2 = transpose ? k : j;
          const unsigned int widx = (type ? index2<n>(idx1,idx2) :
                                     index2<n>(fe_degree-idx1,fe_degree-idx2));

          facevals[j] += constraint_weights[widx]*values[readidx];
        }
      }
      // now can safely write
      for(int j = 0; j < n; ++j) {
        const unsigned int writeidx = direction==0 ? index3<n>(j,i,offset) :
          (direction==1 ? index3<n>(offset,j,i) : index3<n>(offset,i,j));
        values[writeidx] = facevals[j];
      }
    }
  }

  if(constr & SECOND_CONSTR) { // xz-planes
    // now make sure to exclude overlap edge

    const unsigned int offset = (constr & SECOND_TYPE) ? 0 : fe_degree;
    const bool type = constr & THIS_TYPE; // if ==1 face is of 'first' type

    const unsigned int istart = ((constr & FIRST_CONSTR) && (constr & FIRST_TYPE)) ? 1 : 0;
    const unsigned int iend = ((constr & FIRST_CONSTR) && !(constr & FIRST_TYPE)) ? fe_degree : n;

    for(int i = istart; i < iend; ++i) { // loop in z-direction
      for(int j = 0; j < n; ++j) { // loop in x-direction

        facevals[j] = 0;
        for(int k = 0; k < n; ++k) {

          const unsigned int readidx = direction==0 ? index3<n>(k,offset,i) :
            (direction==1 ? index3<n>(i,k,offset) : index3<n>(i,offset,k));

          // if type==0, read reversed
          const unsigned int idx1 = transpose ? j : k;
          const unsigned int idx2 = transpose ? k : j;
          const unsigned int widx = (type ? index2<n>(idx1,idx2) :
                                     index2<n>(fe_degree-idx1,fe_degree-idx2));

          facevals[j] += constraint_weights[widx]*values[readidx];
        }
      }
      // now can safely write
      for(int j = 0; j < n; ++j) {
        const unsigned int writeidx = direction==0 ? index3<n>(j,offset,i) :
          (direction==1 ? index3<n>(i,j,offset) : index3<n>(i,offset,j));
        values[writeidx] = facevals[j];
      }
    }
  }
}


template <int dim, int fe_degree, bool transpose, typename Number>
__device__ void resolve_hanging_nodes_pmem(Number *values, const unsigned int constr)
{
  if(dim==2) {

    const unsigned int n = fe_degree+1;

    if(constr & CONSTR_X) { // x==0 or x==p
      Number facevals[n];

      const bool face = constr & CONSTR_X_TYPE; // if ==1  x==0
      const bool type = constr & CONSTR_Y_TYPE; // if ==1 x-constr is of 'first' type

      Number t;
      const unsigned int offset = face ? 0 : fe_degree;

      for(int i=0; i<=fe_degree; i++) {
        t=0;
        for(int j=0; j<=fe_degree; j++) {
          const unsigned int readidx = index2<n>(offset,j);

          // if type==0, read reversed
          const unsigned int idx1 = transpose ? i : j;
          const unsigned int idx2 = transpose ? j : i;
          const unsigned int widx = (type ? index2<n>(idx1,idx2) :
                                     index2<n>(fe_degree-idx1,fe_degree-idx2));

          t += constraint_weights[widx]*values[readidx];
        }
        facevals[i] = t;
      }

      for(int i = 0; i < n; ++i) {
        const unsigned int writeidx = index2<n>(offset,i);
        values[writeidx] = facevals[i];
      }

    }

    if(constr & CONSTR_Y) { // y==0 or y==p
      Number facevals[n];

      const bool face = constr & CONSTR_Y_TYPE; // if ==1 y==0
      const bool type = constr & CONSTR_X_TYPE; // if ==1 y-constr is of 'first' type

      Number t;
      const unsigned int offset = face ? 0 : fe_degree;

      for(int i=0; i<=fe_degree; i++) {
        t=0;
        for(int j=0; j<=fe_degree; j++) {
          const unsigned int readidx = index2<n>(j,offset);

          // if type==0, read reversed
          const unsigned int idx1 = transpose ? i : j;
          const unsigned int idx2 = transpose ? j : i;
          const unsigned int widx = (type ? index2<n>(idx1,idx2) :
                                     index2<n>(fe_degree-idx1,fe_degree-idx2));

          t += constraint_weights[widx]*values[readidx];
        }
        facevals[i] = t;
      }

      for(int i = 0; i < n; ++i) {
        const unsigned int writeidx = index2<n>(i,offset);
        values[writeidx] = facevals[i];
      }
    }
  }
  else if(dim==3)
  {
    // here we're unfortunately forced to do it like in the shmem case, due to
    // the possibility of shared edge dofs. TODO: maybe do a 'fast path'

    // first x-direction
    interpolate_boundary_3d_pmem<fe_degree,0,transpose> (values, constr);
    // then y-direction
    interpolate_boundary_3d_pmem<fe_degree,1,transpose> (values, constr);
    // now, for a complete surprise, do z-direction!
    interpolate_boundary_3d_pmem<fe_degree,2,transpose> (values, constr);
  }
}


#endif /* __deal2__matrix_free_hanging_nodes_h */