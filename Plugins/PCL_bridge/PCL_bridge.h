#pragma once
#include "Manager.h"


#ifdef __cplusplus
extern "C"{
#endif
    IPerModuleInterface* GetModule();

#ifdef __cplusplus
}
#endif
namespace EagleLib
{
    class PCL_bridge: public Node
    {
    public:
        PCL_bridge();
        virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream &stream = cv::cuda::Stream::Null());
        virtual void Init(bool firstInit);
    };
	class HuMoments : public Node
	{
	public: 
		HuMoments();
		virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream &stream = cv::cuda::Stream::Null());
		virtual void Init(bool firstInit);
	};
}
