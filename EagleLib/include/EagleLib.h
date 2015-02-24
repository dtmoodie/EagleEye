#pragma once

#ifndef RCC_ENABLED
#define RCC_ENABLED
#endif

#ifdef _WIN32
#undef RCC_ENABLED
#else
#define CVAPI_EXPORTS
#endif

#include "nodes/Node.h"	
//#include "nodes/VideoProc/OpticalFlow.h"



/*
 *  Defines the entry point to the image processing library
 *  The core feature of this library is to create a linking framework for each of the subsequent processing nodes.  Very similar to ROS except not through TCP/IP
 *  To do this each node will be represented as an element in a tree.  The top of the tree will commence processing and the results will be converged at the bottom
 *
 *
 *
*/
