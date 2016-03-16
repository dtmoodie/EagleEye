#include "cv_capture.h"
#include "EagleLib/Logging.h"
using namespace EagleLib;


frame_grabber_cv::frame_grabber_cv()
{
    playback_frame_number = -1;
}

bool frame_grabber_cv::LoadFile(const std::string& file_path)
{
    if(d_LoadFile(file_path))
    {
        return true;
    }else
    {
        return h_LoadFile(file_path);
    }
    return false;
}

bool frame_grabber_cv::d_LoadFile(const std::string& file_path)
{
    d_cam.release();
    try
    {
        auto d_temp = cv::cudacodec::createVideoReader(file_path);
        if (d_temp)
        {
            d_cam = d_temp;
            loaded_document = file_path;
            return true;
        }
    }
    catch (cv::Exception& e)
    {

    }
    return false;
}

bool frame_grabber_cv::h_LoadFile(const std::string& file_path)
{
    h_cam.release();
    LOG(info) << "Attemping to load " << file_path;
    boost::mutex::scoped_lock lock(buffer_mtx);
    frame_buffer.clear();
    buffer_begin_frame_number = 0;
    buffer_end_frame_number = 0;
    playback_frame_number = -1;
    try
    {
        h_cam.reset(new cv::VideoCapture());
        if (h_cam)
        {
            if (h_cam->open(file_path))
            {
                loaded_document = file_path;
                playback_frame_number = h_cam->get(cv::CAP_PROP_POS_FRAMES);
                return true;
            }
        }
    }
    catch (cv::Exception& e)
    {
    }
    return false;
}

int frame_grabber_cv::GetNumFrames()
{
    if (d_cam)
    {
        return -1;
    }
    if (h_cam)
    {
        return h_cam->get(cv::CAP_PROP_FRAME_COUNT);
    }
    return -1;
}

/*TS<SyncedMemory> frame_grabber_cv::GetCurrentFrame(cv::cuda::Stream& stream)
{
    return TS<SyncedMemory>(current_frame.timestamp, current_frame.frame_number, current_frame.clone(stream));
}*/

TS<SyncedMemory> frame_grabber_cv::GetFrameImpl(int index, cv::cuda::Stream& stream)
{
    if (d_cam)
    {

    }
    if (h_cam)
    {
        if (h_cam->set(cv::CAP_PROP_POS_FRAMES, index))
        {
            return GetNextFrameImpl(stream);
        }
    }
    return TS<SyncedMemory>();
}

TS<SyncedMemory> frame_grabber_cv::GetNextFrameImpl(cv::cuda::Stream& stream)
{
    if (d_cam)
    {

    }
    if (h_cam)
    {
        cv::Mat h_mat;
        if (h_cam->read(h_mat))
        {
            if (!h_mat.empty())
            {
                cv::cuda::GpuMat d_mat;
                d_mat.upload(h_mat, stream);
                return TS<SyncedMemory>(h_cam->get(cv::CAP_PROP_POS_MSEC), (int)h_cam->get(cv::CAP_PROP_POS_FRAMES), h_mat, d_mat);
            }
        }
    }
    return TS<SyncedMemory>();
}

void frame_grabber_cv::Serialize(ISimpleSerializer* pSerializer)
{
    FrameGrabberBuffered::Serialize(pSerializer);
    SERIALIZE(h_cam);
    SERIALIZE(d_cam);
    //SERIALIZE(current_frame);
}
void frame_grabber_cv::Init(bool firstInit)
{
    FrameGrabberBuffered::Init(firstInit);
}
