#pragma once
#include "EagleLib/nodes/Node.h"
#ifdef __cplusplus
extern "C"{
#endif
    CV_EXPORTS IPerModuleInterface* GetModule();

#ifdef __cplusplus
}
#endif
class MyFreenectDevice;
namespace Freenect
{
	class Freenect;
}

namespace EagleLib
{
namespace Nodes
{
    class camera_freenect: public Node
    {
		Freenect::Freenect* freenect;
		cv::cuda::GpuMat XYZ;
        MyFreenectDevice* myDevice;
        std::vector<uint16_t> depthBuffer;
    public:
        camera_freenect();
		~camera_freenect();
		virtual void Serialize(ISimpleSerializer* pSerializer);
        virtual void Init(bool firstInit);
        virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream &stream);
        virtual bool SkipEmpty() const;
    };
}
}
