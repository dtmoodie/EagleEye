#include <nodes\Node.h>

namespace EagleLib
{
	namespace IO
	{
		class VideoLoader: public Node
		{
		public:
			VideoLoader();
			VideoLoader(std::string file);
			~VideoLoader();
			void loadFile();

			virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat& img);
			bool EOF_reached;

		};
	}
}