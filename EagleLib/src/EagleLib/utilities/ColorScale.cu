#include "EagleLib/utilities/ColorScale.hpp"

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
    if (value > 1.0f)
    {
        if (symmetric) value = 2.0f - value;
        else value = 1.0f;
    }
    if (value < 0) value = 0;
    if (inverted) value = 1.0f - value;
    return value;
}
void ColorScale::Rescale(float alpha, float beta)
{
    start = start * alpha - beta;
    slope *= alpha;
}