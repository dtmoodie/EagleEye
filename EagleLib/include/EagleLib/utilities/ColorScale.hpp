#pragma once
#include "EagleLib/Detail/Export.hpp"
#include <cuda.h>
#include <cuda_runtime.h>

namespace EagleLib
{
    struct EAGLE_EXPORTS ColorScale
    {
	    __host__ __device__ ColorScale(double start_ = 0, double slope_ = 1, bool symmetric_ = false);
	    // Defines where this color starts to take effect, between zero and 1
	    double start;
	    // Defines the slope of increase / decrease for this color between 
	    double slope;
	    // Defines if the slope decreases after it peaks
	    bool	symmetric;
	    // Defines if this color starts high then goes low
	    bool	inverted;
	    bool flipped;
        void Rescale(float alpha, float beta);
	    __host__ __device__ float operator()(float location);
	    __host__ __device__ float GetValue(float location_);
        template<class A> void serialize(A& ar)
        {
            ar(start);
            ar(slope);
            ar(symmetric);
            ar(inverted);
            ar(flipped);
        }
    };
}