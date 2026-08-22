#pragma once

#include "fem/forms/util.hpp"
#include "fem/forms/weights.hpp"

#include "fem/forms/linearforms/linearform.hpp"
#include "fem/forms/linearforms/linear_kernels.hpp"

#include "fem/forms/bilinearforms/bilinearform.hpp"
#include "fem/forms/bilinearforms/bilinear_kernels.hpp"
#include "fem/forms/bilinearforms/dense_linalg.hpp"
#include "fem/forms/bilinearforms/eigen_wrapper.hpp"




///////////////////////////////////////////////////////////////
/// A few common aliases
///////////////////////////////////////////////////////////////
using H1BilinearKernel_S  = GV::H1BilinearKernel<true,false>;	//symetric, non-weighted
using H1BilinearKernel_SW = GV::H1BilinearKernel<true,true>;	//symetric, weighted
using L2BilinearKernel_S  = GV::L2BilinearKernel<true,false>;	//symetric, non-weighted
using L2BilinearKernel_SW = GV::L2BilinearKernel<true,true>;	//symetric, weighted

using L2LinearKernel_W    = GV::L2LinearKernel<true>;			//weighted