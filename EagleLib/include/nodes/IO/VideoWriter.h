#pragma once
#include <nodes/Node.h>
#include <opencv2/videoio.hpp>
#include <opencv2/cudacodec.hpp>

namespace EagleLib
{
	namespace IO
	{
		class CV_EXPORTS VideoWriter : public Node
		{
		public:
			VideoWriter();
			VideoWriter(std::string fileName);
			~VideoWriter();
			cv::cuda::GpuMat doProcess(cv::cuda::GpuMat& img);
			void writeImg(cv::cuda::GpuMat& img);

			bool gpuWriter;
			cv::Ptr<cv::cudacodec::VideoWriter> d_writer;
			cv::VideoWriter						h_writer;
		};
	}
}