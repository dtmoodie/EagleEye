#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

namespace EagleLib
{

    struct EAGLE_EXPORTS ColorScale
    {
	    __host__ __device__ ColorScale(double start_ = 0, double slope_ = 1, bool symmetric_ = false);
	    // Defines where this color starts to take effect, between zero and 1000
	    double start;
	    // Defines the slope of increase / decrease for this color between 1 and 255
	    double slope;
	    // Defines if the slope decreases after it peaks
	    bool	symmetric;
	    // Defines if this color starts high then goes low
	    bool	inverted;
	    bool flipped;
	    __host__ __device__ float operator()(float location);
	    __host__ __device__ float GetValue(float location_);
        template<class A> void serialize(A& ar)
        {
            ar(CEREAL_NVP(start));
            ar(CEREAL_NVP(slope));
            ar(CEREAL_NVP(symmetric));
            ar(CEREAL_NVP(inverted));
            ar(CEREAL_NVP(flipped));
        }
    };
}