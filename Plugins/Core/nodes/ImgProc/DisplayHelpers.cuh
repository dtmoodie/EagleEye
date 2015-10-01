#pragma once
#include "cuda.h"
#include "cuda_runtime.h"
#include <opencv2/core/cuda.hpp>
struct mapper;

struct ColorScale
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
	__host__ __device__ unsigned char operator ()(double location);
	__host__ __device__ unsigned char getValue(double location_);

};

struct color_mapper
{
	// if resolution == -1, calculate the exact mapping every time
	void setMapping(ColorScale& red, ColorScale& green, ColorScale& blue, double min, double max, int resolution);
	void colormap_image(cv::cuda::GpuMat& img, cv::cuda::GpuMat rgb_out);
	mapper* mapper;
};
