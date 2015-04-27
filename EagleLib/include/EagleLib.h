#pragma once

#ifndef RCC_ENABLED
#define RCC_ENABLED
#endif
#define CVAPI_EXPORTS
#ifdef _WIN32
//#undef RCC_ENABLED
#else

#endif
#include "nodes/Node.h"
#include "Manager.h"




/*
 *  Defines the entry point to the image processing library
 *  The core feature of this library is to create a linking framework for each of the subsequent processing nodes.  Very similar to ROS except not through TCP/IP
 *  To do this each node will be represented as an element in a tree.  The top of the tree will commence processing and the results will be converged at the bottom
 *
 *
 *
*/
