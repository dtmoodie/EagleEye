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
    try
    {
        h_cam.reset(new cv::VideoCapture());
        if (h_cam)
        {
            if (h_cam->open(file_path))
            {
                loaded_document = file_path;
                playback_frame_number = h_cam->get(cv::CAP_PROP_POS_FRAMES) + 1;

                return true;
            }
        }
    }
    catch (cv::Exception& e)
    {
    }
    return false;
}

TS<SyncedMemory> frame_grabber_cv::GetFrame(int index, cv::cuda::Stream& stream)
{
    boost::mutex::scoped_lock lock(buffer_mtx);
    for (auto& itr : frame_buffer)
    {
        if (itr.frame_number == index)
        {
            return itr;
        }
    }
    return TS<SyncedMemory>();
}
TS<SyncedMemory> frame_grabber_cv::GetNextFrame(cv::cuda::Stream& stream)
{
    while (frame_buffer.empty())
    {
        //boost::this_thread::interruptible_wait(10);
        boost::this_thread::sleep_for((boost::chrono::milliseconds(10)));
    }
    if (playback_frame_number == -1)
    {
        int min = std::numeric_limits<int>::max();
        {
            boost::mutex::scoped_lock lock(buffer_mtx);
            for (auto itr : frame_buffer)
            {
                min = std::min(min, itr.frame_number);
            }
        }

        playback_frame_number = min;
        auto frame = GetFrame(playback_frame_number, stream);
        if (!frame.empty())
            playback_frame_number++;
        return frame;
    }

    if (update_signal)
        (*update_signal)();
    auto ret = GetFrame(playback_frame_number, stream);
    if (!ret.empty())
        playback_frame_number++;
    return ret;
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

TS<SyncedMemory> frame_grabber_cv::GetCurrentFrame(cv::cuda::Stream& stream)
{
    return TS<SyncedMemory>(current_frame.timestamp, current_frame.frame_number, current_frame.clone(stream));
}

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
                current_frame = TS<SyncedMemory>(h_cam->get(cv::CAP_PROP_POS_MSEC), (int)h_cam->get(cv::CAP_PROP_POS_FRAMES), h_mat, d_mat);
                return TS<SyncedMemory>(current_frame.timestamp, current_frame.frame_number, current_frame.clone(stream));
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
    SERIALIZE(current_frame);
}
void frame_grabber_cv::Init(bool firstInit)
{
    FrameGrabberBuffered::Init(firstInit);
}
