#pragma once
#include "EagleLib/Detail/Export.hpp"

#include "cuda.h"
#include "cuda_runtime.h"

#include <opencv2/core/cuda.hpp>


/*


struct EAGLE_EXPORTS color_mapper
{
	// if resolution == -1, calculate the exact mapping every time
    void setMapping(ColorScale red, ColorScale green, ColorScale blue, double min, double max);
	void colormap_image(cv::cuda::GpuMat& img, cv::cuda::GpuMat& rgb_out, cv::cuda::Stream& stream);

private:
	ColorScale red_, green_, blue_;
	double alpha, beta;
};
*/