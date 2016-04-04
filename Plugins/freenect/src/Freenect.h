#pragma once
#include "EagleLib/frame_grabber_base.h"
#include "EagleLib/ICoordinateManager.h"
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

    class PLUGIN_EXPORTS freenect: public FrameGrabberBuffered
    {
		Freenect::Freenect* _freenect;
		cv::cuda::GpuMat XYZ;
        MyFreenectDevice* _myDevice;

    public:
		class PLUGIN_EXPORTS frame_grabber_freenect_info : public FrameGrabberInfo
		{
		public:
			virtual std::string GetObjectName();
			virtual std::string GetObjectTooltip();
			virtual std::string GetObjectHelp();
			virtual int CanLoadDocument(const std::string& document) const;
			virtual int Priority() const;
			virtual int LoadTimeout() const;
		};
		//freenect();
		~freenect();
		virtual void Serialize(ISimpleSerializer* pSerializer);
		virtual bool LoadFile(const std::string& file_path);
		int GetNumFrames() { return -1; }
		rcc::shared_ptr<ICoordinateManager> GetCoordinateManager();
		virtual TS<SyncedMemory> GetFrameImpl(int index, cv::cuda::Stream& stream);
		virtual TS<SyncedMemory> GetNextFrameImpl(cv::cuda::Stream& stream);
		rcc::shared_ptr<ICoordinateManager> _coordinate_manager;
    };

}
