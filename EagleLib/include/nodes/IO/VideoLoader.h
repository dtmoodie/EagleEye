#include <nodes/Node.h>

#include <opencv2/cudacodec.hpp>
#include <opencv2/videoio.hpp>


namespace EagleLib
{
	namespace IO
	{
		class CV_EXPORTS VideoLoader : public Node
		{
		public:
			VideoLoader();
			~VideoLoader();
			void Init(bool firstInit);
			void loadFile();
            void restartVideo();
            virtual bool SkipEmpty() const;
			virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat& img);
            cv::Ptr<cv::cudacodec::VideoReader> d_videoReader;
            cv::Ptr<cv::VideoCapture>           h_videoReader;
		};
	}
}
