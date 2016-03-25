
#include <EagleLib/nodes/Node.h>
#include <EagleLib/rcc/external_includes/cv_cudafeatures2d.hpp>
#include "EagleLib/utilities/CudaUtils.hpp"
#include <EagleLib/rcc/external_includes/cv_cudafilters.hpp>
#include <EagleLib/rcc/external_includes/cv_cudaoptflow.hpp>
#include <EagleLib/rcc/external_includes/cv_cudaimgproc.hpp>
#include "RuntimeInclude.h"
#include "RuntimeSourceDependency.h"
RUNTIME_COMPILER_SOURCEDEPENDENCY
RUNTIME_MODIFIABLE_INCLUDE

namespace EagleLib
{
	namespace Nodes
	{
		class GoodFeaturesToTrack : public Node
		{
			cv::Ptr<cv::cuda::CornersDetector> detector;
			void update_detector(int depth);
			void detect(const cv::cuda::GpuMat& img, int frame_number, const cv::cuda::GpuMat& mask, cv::cuda::Stream& stream);
		public:
			GoodFeaturesToTrack();
			virtual void Init(bool firstInit);
			virtual TS<SyncedMemory> doProcess(TS<SyncedMemory> img, cv::cuda::Stream& stream = cv::cuda::Stream::Null());
		};

		class FastFeatureDetector : public Node
		{
			cv::Ptr<cv::cuda::Feature2DAsync> detector;

		public:
			FastFeatureDetector();
			virtual void Init(bool firstInit);
			virtual void Serialize(ISimpleSerializer *pSerializer);
			virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat& img, cv::cuda::Stream& stream = cv::cuda::Stream::Null());
		};

		class ORBFeatureDetector : public Node
		{
			cv::Ptr<cv::cuda::ORB> detector;
		public:
			ORBFeatureDetector();
			virtual void Init(bool firstInit);
			virtual void Serialize(ISimpleSerializer *pSerializer);
			virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat& img, cv::cuda::Stream& stream = cv::cuda::Stream::Null());
		};

		class HistogramRange : public Node
		{
			cv::cuda::GpuMat levels;
			void updateLevels(int type);
		public:
			HistogramRange();
			virtual void Init(bool firstInit);
			virtual void Serialize(ISimpleSerializer *pSerializer);
			virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream &stream);
		};

		class CornerHarris : public Node
		{
			cv::Ptr<cv::cuda::CornernessCriteria> detector;
		public:
			CornerHarris();
			virtual void Init(bool firstInit);
			virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream &stream);
		};
		class CornerMinEigenValue : public Node
		{
			cv::Ptr<cv::cuda::CornernessCriteria> detector;
		public:
			CornerMinEigenValue();
			virtual void Init(bool firstInit);
			virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream &stream);
		};
	}
}
