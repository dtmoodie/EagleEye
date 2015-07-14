#include "vlc.h"


using namespace EagleLib;
// callbacks for vlc

void* lock(void* data, void**p_pixels)
{
	std::cout << "lock called" << std::endl;
}
void display(void* data, void* id)
{
	std::cout << "display called" << std::endl;
}
void unlock(void* data, void* id, void* const* p_pixels)
{
	std::cout << "unlock called" << std::endl;
}
void vlcCamera::Init(bool firstInit)
{
	vlcInstance = nullptr;
	mp = nullptr;
	media = nullptr;
	RTSPCamera::Init(firstInit);
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
void vlcCamera::onSourceChange()
{
	const char* const vlc_args[] = { "-I", "dummy", "--ignore-config", "--extraintf=logger", "--verbose=2" };
	vlcInstance = libvlc_new(sizeof(vlc_args) / sizeof(vlc_args[0]), vlc_args);
	media = libvlc_media_new_path(vlcInstance, getParameter<std::string>("Source")->Data()->c_str());
	mp = libvlc_media_player_new_from_media(media);
	libvlc_media_release(media);
	libvlc_video_set_callbacks(mp, lock, unlock, display, this);
	libvlc_video_set_format(mp, "RV24", 1920, 1080, 1920 * 3);
	int height = libvlc_video_get_height(mp);
	int width = libvlc_video_get_width(mp);


}

cv::cuda::GpuMat vlcCamera::doProcess(cv::cuda::GpuMat& img, cv::cuda::Stream& stream)
{

	return img;
}
