#include "DisplayHelpers.cuh"



struct LUT_Mapper
{
	LUT_Mapper(double max, double min, ColorScale& red, ColorScale& blue, ColorScale& green);
	
};



__host__ __device__ ColorScale::ColorScale(double start_, double slope_, bool symmetric_)
{
	start = start_;
	slope = slope_;
	symmetric = symmetric_;
	flipped = false;
	inverted = false;
}
unsigned char __host__ __device__  ColorScale::operator ()(double location)
{
	return getValue(location);
}

unsigned char __host__ __device__  ColorScale::getValue(double location_)
{
	double value = 0;
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
	return (unsigned char)value;
}
void color_mapper::setMapping(ColorScale& red, ColorScale& green, ColorScale& blue, double min, double max, int resolution)
{

}

void color_mapper::colormap_image(cv::cuda::GpuMat& img, cv::cuda::GpuMat rgb_out)
{
	rgb_out.create(img.size(), CV_8UC3);
	CV_Assert(img.channels() == 1);




}