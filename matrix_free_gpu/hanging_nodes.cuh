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

// Here is the system for how we store constraint types in a binary mask. This
// is not a complete contradiction-free system; i.e. there are invalid states
// which we just assume that we never get.

// If the mask is == 0, there are no constraints whatsoever. Then there are
// three different fields with one bit per dimension.  The first field
// determines the type, or the position of an element along each direction.  The
// second field determines if there is a constrained face with that direction as
// normal.  The last field determines if there is a constrained edge of a given
// pair of coordinate planes, but where neither of the corresponding faces are
// constrained (only valid in 3D).

// The element is placed in the 'first position' along *-axis.
// these also determine which face is constrained. For example, in 2D, if
// CONSTR_FACE_X and CONSTR_TYPE_X is set, then x==0 is constrained.
#define CONSTR_TYPE_X  (1<<0)
#define CONSTR_TYPE_Y  (1<<1)
#define CONSTR_TYPE_Z  (1<<2)

// element has a constraint at *==0 or *==p face
#define CONSTR_FACE_X  (1<<3)
#define CONSTR_FACE_Y  (1<<4)
#define CONSTR_FACE_Z  (1<<5)

// element has a constraint at *==0 or *==p edge
#define CONSTR_EDGE_XY  (1<<6)
#define CONSTR_EDGE_YZ  (1<<7)
#define CONSTR_EDGE_ZX  (1<<8)

#define NOTRANSPOSE false
#define TRANSPOSE true

//=============================================================================
// setup functions
//=============================================================================

template <int dim>
class HangingNodes {
private:
  typedef typename DoFHandler<dim>::cell_iterator cell_iterator;
  typedef typename DoFHandler<dim>::active_cell_iterator active_cell_iterator;
  const unsigned int n_raw_lines;
  std::vector<std::vector<std::pair<cell_iterator,unsigned int> > > line_to_cells;
  const std::vector<unsigned int> &lex_mapping;
  const unsigned int fe_degree;
  const DoFHandler<dim> &dof_handler;

public:
  HangingNodes(unsigned int fe_degree, const DoFHandler<dim> &dof_handler,
               const std::vector<unsigned int> &lex_mapping)
    : n_raw_lines(dof_handler.get_tria().n_raw_lines()),
      line_to_cells(dim==3?n_raw_lines:0),
      dof_handler(dof_handler),
      lex_mapping(lex_mapping),
      fe_degree(fe_degree)
  {
    // setup line-to-cell mapping for edge constraints
    setup_line_to_cell(); // does nothing for dim==2

    setup_constraint_weights();
  }


  template <typename T>
  void setup_constraints(unsigned int &mask,
                         std::vector<unsigned int> &dof_indices,
                         const T &cell, unsigned int cellid) const;
private:

  // helper functions
  void setup_line_to_cell();

  static void rotate_subface_index(unsigned int &subface_index, int times);

  static void rotate_face(std::vector<types::global_dof_index> &dofs, int times,
                          unsigned int n_dofs_1d);

  static unsigned int line_dof_idx(unsigned int local_line, unsigned int dof,
                                   unsigned int n_dofs_1d);

  static void transpose_face(std::vector<types::global_dof_index> &dofs,
                             unsigned int n_dofs_1d);

  static void transpose_subface_index(unsigned int &subface);

  void setup_constraint_weights();

};

// helper function
template <int dim>
void get_lex_face_mapping(std::vector<types::global_dof_index> &mapping,
                          const unsigned int p);


template <int dim>
void HangingNodes<dim>::setup_line_to_cell()
{
  // In 3D, we can have DoFs on only an edge be constrained (e.g. in a cartesian
  // 2x2x2 grid, where only the upper left 2 cells are refined. This sets up a
  // helper data structure in the form of a mapping from edges (i.e. lines) to
  // neighboring cells. since this is only relevant for 3D, do nothing otherwise

  if(dim==3) {
    // mapping from an edge to which children that share that edge.
    const unsigned int line_to_children[12][2] = {{0,2},
                                                  {1,3},
                                                  {0,1},
                                                  {2,3},
                                                  {4,6},
                                                  {5,7},
                                                  {4,5},
                                                  {6,7},
                                                  {0,4},
                                                  {1,5},
                                                  {2,6},
                                                  {3,7}};

    std::vector<std::vector<std::pair<cell_iterator,unsigned int> > >
      line_to_inactive_cells(n_raw_lines);

    // First add active AND inactive cells to their lines:
    cell_iterator
      cell = dof_handler.begin(),
      endc = dof_handler.end();
    for (; cell!=endc; ++cell) {

      for (unsigned int line=0; line<GeometryInfo<dim>::lines_per_cell; ++line) {

        const unsigned int line_idx = cell->line(line)->index();

        if(cell->active()) {
          line_to_cells[line_idx].push_back(std::make_pair(cell,line));
        }
        else {
          line_to_inactive_cells[line_idx].push_back(std::make_pair(cell,line));
        }
      }
    }

    // now, we can access edge-neighboring active cells on same level

    // to also access those on HIGHER level, i.e. coarser cells, we should add all
    // active cells of an edge to the edges "children". These are found from
    // looking at the (corresponding) edge of children of inactive edge neighbors.


    for(unsigned int line_idx=0; line_idx<n_raw_lines; line_idx++) {

      if(line_to_cells[line_idx].size() > 0 &&
         line_to_inactive_cells[line_idx].size() > 0) {
        // we now have cells to add (active ones), and edges to which they should
        // be added (inactive cells).

        // we only really need to consider ONE of the inactive cells, since the
        // edge children are all the same for all of them. We also know that since
        // they have active edge neighbors, they will have active children. Now test this:

        if(true) {
          for(auto cl : line_to_inactive_cells[line_idx])
            for(int i=0; i< cl.first->n_children(); ++i)
              if(!cl.first->child(i)->active()) {
                fprintf(stderr,"Internal error: Children of cell with active edge-neighbor must be active!\n");
                ExcInternalError();
              }
        }


        const cell_iterator& inactive_cell = line_to_inactive_cells[line_idx][0].first;
        const unsigned int neighbor_line = line_to_inactive_cells[line_idx][0].second;

        for(int c=0; c<2; ++c) {
          const cell_iterator &child = inactive_cell->child(line_to_children[neighbor_line][c]);

          const unsigned int child_line_idx = child->line(neighbor_line)->index();

          // now add ALL active cells
          for(auto cl : line_to_cells[line_idx])
            line_to_cells[child_line_idx].push_back(cl);
        }
      }
    }

  }
}


template <int dim>
template <typename T>
void HangingNodes<dim>::setup_constraints(unsigned int &mask,
                                          std::vector<unsigned int> &dof_indices,
                                          const T &cell, unsigned int cellid) const
{
  mask = 0;
  const unsigned int p = fe_degree;
  const unsigned int n_dofs_1d = fe_degree+1;
  const unsigned int dofs_per_face = ipowf(n_dofs_1d,dim-1);

  std::vector<types::global_dof_index> neighbor_dofs(dofs_per_face);

  std::vector<unsigned int> lex_face_mapping;
  get_lex_face_mapping<dim> (lex_face_mapping,p);

  for (unsigned int face=0; face<GeometryInfo<dim>::faces_per_cell; ++face) {
    const active_cell_iterator &
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
          mask |= CONSTR_FACE_X;
          if(face==0)
            mask |= CONSTR_TYPE_X;
          if(subface==0)
            mask |= CONSTR_TYPE_Y;
        }
        else {
          mask |= CONSTR_FACE_Y;
          if(face==2)
            mask |= CONSTR_TYPE_Y;
          if(subface==0)
            mask |= CONSTR_TYPE_X;
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
          mask |= CONSTR_FACE_X;
          if(face==0)
            mask |= CONSTR_TYPE_X;
          if(subface%2==0)
            mask |= CONSTR_TYPE_Y;
          if(subface/2==0)
            mask |= CONSTR_TYPE_Z;
        }
        // XZ-plane
        else if(face < 4) {
          mask |= CONSTR_FACE_Y;
          if(face==2)
            mask |= CONSTR_TYPE_Y;
          if(subface%2==0)
            mask |= CONSTR_TYPE_Z;
          if(subface/2==0)
            mask |= CONSTR_TYPE_X;
        }
        // XY-plane
        else {
          mask |= CONSTR_FACE_Z;
          if(face==4)
            mask |= CONSTR_TYPE_Z;
          if(subface%2==0)
            mask |= CONSTR_TYPE_X;
          if(subface/2==0)
            mask |= CONSTR_TYPE_Y;
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

  // In 3D we can have a situation where only DoFs on an edge are
  // constrained. Append these here.
  if(dim==3) {

    // for each line on cell, which faces does it belong to, what is the edge
    // mask, what is the types of the faces it belong to, and what is the type
    // along the edge?
    const unsigned int line_to_edge[12][4] =
      {{CONSTR_FACE_X|CONSTR_FACE_Z,CONSTR_EDGE_ZX,CONSTR_TYPE_X|CONSTR_TYPE_Z,CONSTR_TYPE_Y},
       {CONSTR_FACE_X|CONSTR_FACE_Z,CONSTR_EDGE_ZX,CONSTR_TYPE_Z,              CONSTR_TYPE_Y},
       {CONSTR_FACE_Y|CONSTR_FACE_Z,CONSTR_EDGE_YZ,CONSTR_TYPE_Y|CONSTR_TYPE_Z,CONSTR_TYPE_X},
       {CONSTR_FACE_Y|CONSTR_FACE_Z,CONSTR_EDGE_YZ,CONSTR_TYPE_Z,              CONSTR_TYPE_X},
       {CONSTR_FACE_X|CONSTR_FACE_Z,CONSTR_EDGE_ZX,CONSTR_TYPE_X,              CONSTR_TYPE_Y},
       {CONSTR_FACE_X|CONSTR_FACE_Z,CONSTR_EDGE_ZX,            0,              CONSTR_TYPE_Y},
       {CONSTR_FACE_Y|CONSTR_FACE_Z,CONSTR_EDGE_YZ,CONSTR_TYPE_Y,              CONSTR_TYPE_X},
       {CONSTR_FACE_Y|CONSTR_FACE_Z,CONSTR_EDGE_YZ,            0,              CONSTR_TYPE_X},
       {CONSTR_FACE_X|CONSTR_FACE_Y,CONSTR_EDGE_XY,CONSTR_TYPE_X|CONSTR_TYPE_Y,CONSTR_TYPE_Z},
       {CONSTR_FACE_X|CONSTR_FACE_Y,CONSTR_EDGE_XY,CONSTR_TYPE_Y,              CONSTR_TYPE_Z},
       {CONSTR_FACE_X|CONSTR_FACE_Y,CONSTR_EDGE_XY,CONSTR_TYPE_X,              CONSTR_TYPE_Z},
       {CONSTR_FACE_X|CONSTR_FACE_Y,CONSTR_EDGE_XY,            0,              CONSTR_TYPE_Z}};

    // for each edge of this cell ...
    for (unsigned int local_line=0; local_line<GeometryInfo<dim>::lines_per_cell; ++local_line) {

      // that we don't already have a constraint for as part of a face...
      if(!(mask & line_to_edge[local_line][0])) {

        // for each all cells which share that edge...
        const unsigned int line = cell->line(local_line)->index();

        for(auto edge_neighbor : line_to_cells[line]) {
          // if one of them is coarser than us ...
          const cell_iterator neighbor_cell = edge_neighbor.first;

          if(neighbor_cell->level() < cell->level()) {
            // edge is constrained!
            const unsigned int local_line_neighbor = edge_neighbor.second;

            mask |= line_to_edge[local_line][1] | line_to_edge[local_line][2];
            // add the 'type' of the line

            bool flipped = false;
            if(cell->line(local_line)->vertex_index(0) == neighbor_cell->line(local_line_neighbor)->vertex_index(0)) {
              // assuming line directions matches axes directions
              // we have an unflipped edge of first type
              mask |= line_to_edge[local_line][3];


            }
            else if(cell->line(local_line)->vertex_index(1) ==
                    neighbor_cell->line(local_line_neighbor)->vertex_index(1)) {
              // we have an unflipped edge of second type
            }
            else if(cell->line(local_line)->vertex_index(1) ==
                    neighbor_cell->line(local_line_neighbor)->vertex_index(0)) {
              // we have a flipped edge of second type
              flipped = true;
            }
            else if(cell->line(local_line)->vertex_index(0) ==
                    neighbor_cell->line(local_line_neighbor)->vertex_index(1)) {
              // we have a flipped edge of first type
              mask |= line_to_edge[local_line][3];
              flipped = true;
            }
            else {
              fprintf(stderr,"Internal error -- failed to match own edge line with that of neighbor\n");
              ExcInternalError();
              // ERROR! shouldn't happen!
            }

            // copy the unconstrained values
            neighbor_dofs.resize(n_dofs_1d*n_dofs_1d*n_dofs_1d);
            neighbor_cell->get_dof_indices(neighbor_dofs);

            for(int i = 0; i < n_dofs_1d; ++i) {
              // get local dof index along line
              const unsigned int idx = line_dof_idx(local_line,i,n_dofs_1d);

              dof_indices[idx] = neighbor_dofs[lex_mapping[line_dof_idx(local_line_neighbor,
                                                                        flipped ? p-i : i,
                                                                        n_dofs_1d)]];
            }

            break; // stop looping over edge neighbors

          } // level difference if statement
        } // loop over edge neighbors
      } // check for no existing face constraints
    } // loop over edges of cell
  } // dim==3
}


template <int dim>
void get_lex_face_mapping(std::vector<types::global_dof_index> &mapping,
                          const unsigned int p) {
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

template <int dim>
void HangingNodes<dim>::rotate_subface_index(unsigned int &subface_index, int times)
{
  const unsigned int rot_mapping[4] = {2, 0, 3, 1};

  times = times % 4;
  times = times < 0 ? times + 4 : times;
  for(int t = 0; t < times; ++t) {
    subface_index = rot_mapping[subface_index];
  }
}


template <int dim>
void HangingNodes<dim>::rotate_face(std::vector<types::global_dof_index> &dofs, int times, unsigned int n_dofs_1d)
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

template <int dim>
unsigned int HangingNodes<dim>::line_dof_idx(unsigned int local_line, unsigned int dof,
                                                    unsigned int n_dofs_1d)
{
  const unsigned int p = n_dofs_1d-1;
  unsigned int x,y,z;

  if(local_line < 8) {
    x = (local_line%4 == 0) ? 0 : (local_line%4 == 1) ? p : dof;
    y = (local_line%4 == 2) ? 0 : (local_line%4 == 3) ? p : dof;
    z = (local_line/4)*p;
  }
  else {
    x=((local_line-8)%2)*p;
    y=((local_line-8)/2)*p;
    z=dof;
  }

  return n_dofs_1d*n_dofs_1d*z + n_dofs_1d*y + x;
}
template <int dim>
void HangingNodes<dim>::transpose_face(std::vector<types::global_dof_index> &dofs, unsigned int n_dofs_1d)
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

template <int dim>
void HangingNodes<dim>::transpose_subface_index(unsigned int &subface)
{
  if(subface == 1) subface = 2;
  else if(subface == 2) subface = 1;
}

template <int dim>
void HangingNodes<dim>::setup_constraint_weights() {
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

  const unsigned int this_type  = (direction==0) ? CONSTR_TYPE_X :
    (direction==1) ? CONSTR_TYPE_Y : CONSTR_TYPE_Z;
  const unsigned int face1_type = (direction==0) ? CONSTR_TYPE_Y :
    (direction==1) ? CONSTR_TYPE_Z : CONSTR_TYPE_X;
  const unsigned int face2_type = (direction==0) ? CONSTR_TYPE_Z :
    (direction==1) ? CONSTR_TYPE_X : CONSTR_TYPE_Y;

  // if computing in x-direction, need to match against CONSTR_FACE_Y or CONSTR_FACE_Z
  const unsigned int face1 = (direction==0) ? CONSTR_FACE_Y :
    (direction==1) ? CONSTR_FACE_Z : CONSTR_FACE_X;
  const unsigned int face2 = (direction==0) ? CONSTR_FACE_Z :
    (direction==1) ? CONSTR_FACE_X : CONSTR_FACE_Y;
  const unsigned int edge  = (direction==0) ? CONSTR_EDGE_YZ :
    (direction==1) ? CONSTR_EDGE_ZX : CONSTR_EDGE_XY;

  if( constr & (face1|face2|edge)) {
    const unsigned int interp_idx = (direction==0) ? xidx : (direction==1) ? yidx : zidx;
    const unsigned int face1_idx  = (direction==0) ? yidx : (direction==1) ? zidx : xidx;
    const unsigned int face2_idx  = (direction==0) ? zidx : (direction==1) ? xidx : yidx;

    syncthreads();

    Number t = 0;

    const bool on_face1 = (constr & face1_type) ? (face1_idx==0) : (face1_idx==fe_degree);
    const bool on_face2 = (constr & face2_type) ? (face2_idx==0) : (face2_idx==fe_degree);
    const bool flag = ( ((constr & face1) && on_face1) || ((constr & face2) && on_face2) ||
                        ((constr & edge) && on_face1 && on_face2) );

    if(flag) {
      const bool type = constr & this_type;

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

  const unsigned int this_type = (direction==0) ? CONSTR_TYPE_X : CONSTR_TYPE_Y;

  if(constr &
     (((direction==0) ? CONSTR_FACE_Y : 0) |
      ((direction==1) ? CONSTR_FACE_X : 0))
     ) {

    const unsigned int interp_idx = (direction==0) ? xidx : yidx;

    syncthreads();

    Number t = 0;
    const bool flag =
      ((direction==0) && ((constr & CONSTR_TYPE_Y) ? (yidx==0) : (yidx==fe_degree))) ||
      ((direction==1) && ((constr & CONSTR_TYPE_X) ? (xidx==0) : (xidx==fe_degree)));

    if(flag) {
      const bool type = constr & this_type;

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

  const unsigned int this_type  = (direction==0) ? CONSTR_TYPE_X :
    (direction==1) ? CONSTR_TYPE_Y : CONSTR_TYPE_Z;
  const unsigned int face1_type = (direction==0) ? CONSTR_TYPE_Z :
    (direction==1) ? CONSTR_TYPE_X : CONSTR_TYPE_Y;
  const unsigned int face2_type = (direction==0) ? CONSTR_TYPE_Y :
    (direction==1) ? CONSTR_TYPE_Z : CONSTR_TYPE_Z;

  const unsigned int face1 = (direction==0) ? CONSTR_FACE_Z :
    (direction==1) ? CONSTR_FACE_X : CONSTR_FACE_Y;
  const unsigned int face2 = (direction==0) ? CONSTR_FACE_Y :
    (direction==1) ? CONSTR_FACE_Z : CONSTR_FACE_X;
  const unsigned int edge =  (direction==0) ? CONSTR_EDGE_YZ :
    (direction==1) ? CONSTR_EDGE_ZX : CONSTR_EDGE_XY;

  Number facevals[n];

  // We make sure to exclude overlap edge, and compute it separately afterwards,
  // to include the case when only the edge is constrained.

  if(constr & face1) { // first face (xy/yz/zx - plane)

    const unsigned int offset = (constr & face1_type) ? 0 : fe_degree;
    const bool type = constr & this_type; // if ==1 face is of 'first' type

    const unsigned int istart = ((constr & face2) && (constr & face2_type)) ? 1 : 0;
    const unsigned int iend = ((constr & face2) && !(constr & face2_type)) ? fe_degree : n;

    for(int i = istart; i < iend; ++i) { // loop perpendicular to constraint
      for(int j = 0; j < n; ++j) { // loop along constraint

        facevals[j] = 0;
        for(int k = 0; k < n; ++k) {
          const unsigned int readidx = (direction==0) ? index3<n>(k,i,offset) :
            (direction==1) ? index3<n>(offset,k,i) : index3<n>(i,offset,k);

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
        const unsigned int writeidx = (direction==0) ? index3<n>(j,i,offset) :
          (direction==1) ? index3<n>(offset,j,i) : index3<n>(i,offset,j);
        values[writeidx] = facevals[j];
      }
    }
  }

  if(constr & face2) { // second face (xz/yx/zy - plane)

    const unsigned int offset = (constr & face2_type) ? 0 : fe_degree;
    const bool type = constr & this_type; // if ==1 face is of 'first' type

    const unsigned int istart = ((constr & face1) && (constr & face1_type)) ? 1 : 0;
    const unsigned int iend = ((constr & face1) && !(constr & face1_type)) ? fe_degree : n;

    for(int i = istart; i < iend; ++i) { // loop perpendicular to constraint
      for(int j = 0; j < n; ++j) { // loop along constraint

        facevals[j] = 0;
        for(int k = 0; k < n; ++k) {

          const unsigned int readidx = (direction==0) ? index3<n>(k,offset,i) :
            (direction==1) ? index3<n>(i,k,offset) : index3<n>(offset,i,k);

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
        const unsigned int writeidx = (direction==0) ? index3<n>(j,offset,i) :
          (direction==1) ? index3<n>(i,j,offset) : index3<n>(offset,i,j);
        values[writeidx] = facevals[j];
      }
    }
  }

  // compute edge
  if( ((constr&face1) && (constr&face1))
      || (constr&edge)) {

    const unsigned int offset1 = (constr & face2_type) ? 0 : fe_degree;
    const unsigned int offset2 = (constr & face1_type) ? 0 : fe_degree;

    const bool type = constr & this_type; // if ==1 face is of 'first' type

    for(int j = 0; j < n; ++j) { // loop along constraint

      facevals[j] = 0;
      for(int k = 0; k < n; ++k) {
        const unsigned int readidx = (direction==0) ? index3<n>(k,offset1,offset2) :
          (direction==1) ? index3<n>(offset2,k,offset1) : index3<n>(offset1,offset2,k);

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
      const unsigned int writeidx = (direction==0) ? index3<n>(j,offset1,offset2) :
        (direction==1) ? index3<n>(offset2,j,offset1) : index3<n>(offset1,offset2,j);
      values[writeidx] = facevals[j];
    }
  }
}


template <int dim, int fe_degree, bool transpose, typename Number>
__device__ void resolve_hanging_nodes_pmem(Number *values, const unsigned int constr)
{
  if(dim==2) {

    const unsigned int n = fe_degree+1;

    if(constr & CONSTR_FACE_X) { // x==0 or x==p
      Number facevals[n];

      const bool face = constr & CONSTR_TYPE_X; // if ==1  x==0
      const bool type = constr & CONSTR_TYPE_Y; // if ==1 x-constr is of 'first' type

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

    if(constr & CONSTR_FACE_Y) { // y==0 or y==p
      Number facevals[n];

      const bool face = constr & CONSTR_TYPE_Y; // if ==1 y==0
      const bool type = constr & CONSTR_TYPE_X; // if ==1 y-constr is of 'first' type

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