#ifndef CONSTANTS_H
#define CONSTANTS_H

#include <climits>

#define MIN_ASSEMBLY_NUMBER_OF_PARTICLES 0
#define SOLID_PHASE_MARKER 0 //do not change. negative markers for isolated pores, positive markers for pores connected to boundary
#define DEFAULT_FLUID_PHASE_MARKER 1
#define UNDEFINED_MARKER INT_MIN

//MAC defaults
#define MAC_DOMAIN_MARKER DEFAULT_FLUID_PHASE_MARKER
#define MAC_DEFAULT_TOL 1E-6
#define MAC_DEFAULT_MAX_OUTER_ITERATIONS 100
#define MAC_MULTIGRID_MIN_DIMENSION 32


#endif