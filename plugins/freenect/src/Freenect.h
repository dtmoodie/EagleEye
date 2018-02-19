#pragma once
#include <Aquila/framegrabbers/IFrameGrabber.hpp>
#include <Aquila/types/SyncedMemory.hpp>

class MyFreenectDevice;
namespace Freenect
{
    class Freenect;
}

namespace aq
{
    class frame_grabber_freenect_info : public FrameGrabberInfo
    {
    public:
        virtual std::string GetObjectName();
        virtual std::string GetObjectTooltip();
        virtual std::string GetObjectHelp();
        virtual int CanLoadDocument(const std::string& document) const;
        virtual int Priority() const;
        virtual int LoadTimeout() const;
    };

    class freenect: public nodes::IFrameGrabber
    {
        Freenect::Freenect* _freenect;
        cv::cuda::GpuMat XYZ;
        MyFreenectDevice* _myDevice;

    public:
        ~freenect();
        virtual void Serialize(ISimpleSerializer* pSerializer);
        virtual bool loadData(std::string file_path);
    };

}
