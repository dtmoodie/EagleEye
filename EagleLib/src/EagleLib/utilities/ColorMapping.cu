#include "ColorMapping.hpp"

using namespace EagleLib;

__host__ __device__ ColorScale::ColorScale(double start_, double slope_, bool symmetric_)
{
    start = start_;
    slope = slope_;
    symmetric = symmetric_;
    flipped = false;
    inverted = false;
}
float __host__ __device__  ColorScale::operator ()(float location)
{
    return GetValue(location);
}

float __host__ __device__  ColorScale::GetValue(float location_)
{
    float value = 0;
    if (location_ > start)
    {
        value = (location_ - start)*slope;
    }
    else
    { 
        value = 0;
    }
    if (value > 255)
    {
        if (symmetric) value = 512 - value;
        else value = 255;
    }
    if (value < 0) value = 0;
    if (inverted) value = 255 - value;
    return value;
}