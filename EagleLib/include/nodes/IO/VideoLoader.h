#include <nodes/Node.h>


#if _WIN32
#include <opencv2/cudacodec.hpp>
#else
#include <opencv2/videoio.hpp>
#endif

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

			virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat& img);
#if _WIN32
			cv::Ptr<cv::cudacodec::VideoReader> videoReader;
#else

#endif
		};
	}
}
