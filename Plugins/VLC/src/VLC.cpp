#include "VLC.h"
#include "Aquila/framegrabbers/FrameGrabberInfo.hpp"
#include "Aquila/nodes/Node.hpp"

using namespace aq;
using namespace aq::nodes;

void* lock(void* data, void** p_pixels)
{
    // std::cout << "lock called" << std::endl;
    vlcCamera* node = static_cast<vlcCamera*>(data);
    cv::Mat img;
    img.create(1080, 1920, CV_8UC3);
    *p_pixels = img.data;
    node->img_queue.enqueue(img);
    return nullptr;
}

void display(void* data, void* id)
{
    // std::cout << "display called" << std::endl;
}

void unlock(void* data, void* id, void* const* p_pixels)
{
    // std::cout << "unlock called" << std::endl;
}

void vlcCamera::nodeInit(bool firstInit)
{
    vlcInstance = nullptr;
    mp = nullptr;
    media = nullptr;
}

bool vlcCamera::Load(std::string file)
{
    const char* const vlc_args[] = {"-I", "dummy", "--ignore-config", "--extraintf=logger", "--verbose=2"};
    vlcInstance = libvlc_new(sizeof(vlc_args) / sizeof(vlc_args[0]), vlc_args);
    if (vlcInstance == nullptr)
    {
        MO_LOG(error) << "vlcInstance == nullptr";
        return false;
    }

    media = libvlc_media_new_location(vlcInstance, file.c_str());
    if (media == nullptr)
    {
        MO_LOG(error) << "media == nullptr";
        return false;
    }

    mp = libvlc_media_player_new_from_media(media);
    if (mp == nullptr)
    {
        MO_LOG(error) << "mp == nullptr";
        return false;
    }
    libvlc_media_release(media);
    libvlc_video_set_callbacks(mp, lock, unlock, display, this);
    libvlc_video_set_format(mp, "RV24", 1920, 1080, 1920 * 3);
    MO_LOG(info) << "Source setup correctly";

    int height = libvlc_video_get_height(mp);
    int width = libvlc_video_get_width(mp);
    return true;
}

vlcCamera::~vlcCamera()
{
    if (mp)
    {
        libvlc_media_player_stop(mp);
        libvlc_media_player_release(mp);
    }
    if (media)
    {
        libvlc_media_release(media);
    }
    if (vlcInstance)
    {
    }
}

bool vlcCamera::processImpl()
{
    cv::Mat img;
    if (img_queue.try_dequeue(img))
    {
        image_param.updateData(img);
    }
    return true;
}
MO_REGISTER_CLASS(vlcCamera);
