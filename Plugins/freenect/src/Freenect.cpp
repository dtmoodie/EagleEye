#include "Freenect.h"
#include "libfreenect/libfreenect.hpp"

IPerModuleInterface* GetModule()
{
    return PerModuleInterface::GetInstance();
}

class MyFreenectDevice : public Freenect::FreenectDevice
{
public:
    MyFreenectDevice(freenect_context *_ctx, int _index)
        : Freenect::FreenectDevice(_ctx, _index),
          m_buffer_video(freenect_find_video_mode(FREENECT_RESOLUTION_MEDIUM, FREENECT_VIDEO_RGB).bytes),
          m_buffer_depth(freenect_find_depth_mode(FREENECT_RESOLUTION_MEDIUM, FREENECT_DEPTH_REGISTERED).bytes / 2),
          m_new_rgb_frame(false), m_new_depth_frame(false)
    {
        setDepthFormat(FREENECT_DEPTH_REGISTERED);
    }

    // Do not call directly, even in child
    void VideoCallback(void *_rgb, uint32_t timestamp)
    {
        std::lock_guard<std::mutex> lock(m_rgb_mutex);
        uint8_t* rgb = static_cast<uint8_t*>(_rgb);
        copy(rgb, rgb+getVideoBufferSize(), m_buffer_video.begin());
        m_new_rgb_frame = true;
    }

    // Do not call directly, even in child
    void DepthCallback(void *_depth, uint32_t timestamp)
    {
        std::lock_guard<std::mutex> lock(m_depth_mutex);
        uint16_t* depth = static_cast<uint16_t*>(_depth);
        copy(depth, depth+getDepthBufferSize()/2, m_buffer_depth.begin());
        m_new_depth_frame = true;
    }

    bool getRGB(std::vector<uint8_t> &buffer)
    {
        std::lock_guard<std::mutex> lock(m_rgb_mutex);

        if (!m_new_rgb_frame)
            return false;

        buffer.swap(m_buffer_video);
        m_new_rgb_frame = false;

        return true;
    }

    bool getDepth(std::vector<uint16_t> &buffer)
    {
        std::lock_guard<std::mutex> lock(m_depth_mutex);

        if (!m_new_depth_frame)
            return false;

        buffer.swap(m_buffer_depth);
        m_new_depth_frame = false;

        return true;
    }

private:
    std::mutex m_rgb_mutex;
    std::mutex m_depth_mutex;
    std::vector<uint8_t> m_buffer_video;
    std::vector<uint16_t> m_buffer_depth;
    bool m_new_rgb_frame;
    bool m_new_depth_frame;
};

Freenect::Freenect freenect;
using namespace EagleLib;
void camera_freenect::Init(bool firstInit)
{
    myDevice = &freenect.createDevice<MyFreenectDevice>(0);
    myDevice->startVideo();
    myDevice->startDepth();
    depthBuffer.resize(640*480);
}

cv::cuda::GpuMat camera_freenect::doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream &stream)
{
    if(myDevice->getDepth(depthBuffer))
    {
        cv::Mat h_depth(640,480, CV_16U, (void*)&depthBuffer[0]);
        img.upload(h_depth, stream);
    }
    return cv::cuda::GpuMat();
}

bool camera_freenect::SkipEmpty() const
{
    return false;
}

NODE_DEFAULT_CONSTRUCTOR_IMPL(camera_freenect)
