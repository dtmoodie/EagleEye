#include "vlc.h"


using namespace EagleLib;
// callbacks for vlc

void SetupIncludes()
{
#ifdef PROJECT_INCLUDES
{
	std::string str(PROJECT_INCLUDES);
	std::string::size_type pos = str.find_first_of('+');
	std::string::size_type prevPos = 0;
	while (pos != std::string::npos)
	{
		EagleLib::NodeManager::getInstance().addIncludeDir(str.substr(prevPos, pos - prevPos).c_str());
		prevPos = pos + 1;
		pos = str.find_first_of('+', pos + 1);
	}
}
#endif 
#ifdef PROJECT_LIB_DIRS
{
	std::string str(PROJECT_LIB_DIRS);
	std::string::size_type pos = str.find_first_of('+');
	std::string::size_type prevPos = 0;
	while (pos != std::string::npos)
	{
		EagleLib::NodeManager::getInstance().addLinkDir(str.substr(prevPos, pos - prevPos).c_str());
		prevPos = pos + 1;
		pos = str.find_first_of('+', pos + 1);
	}
}
#endif
}
IPerModuleInterface* GetModule()
{
	return PerModuleInterface::GetInstance();
}



void* lock(void* data, void**p_pixels)
{
	std::cout << "lock called" << std::endl;
	vlcCamera* node = static_cast<vlcCamera*>(data);
	cv::cuda::HostMem* dest = node->h_dest.getFront();
	dest->create(1080,1920, CV_8UC3);
	*p_pixels = dest->data;
	node->imgQueue.push(dest);
	return nullptr;
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
	updateParameter<std::string>("Source", "rtsp://192.168.1.152/axis-media/media.amp");
	RegisterParameterCallback("Source", boost::bind(&vlcCamera::onSourceChange, this));
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
	if (vlcInstance == nullptr)
		return log(Error, "vlcInstance == nullptr");
	media = libvlc_media_new_location(vlcInstance, getParameter<std::string>("Source")->Data()->c_str());
	if (media == nullptr)
		return log(Error, "media == nullptr");
	mp = libvlc_media_player_new_from_media(media);
	if (mp == nullptr)
		return log(Error, "mp == nullptr");
	libvlc_media_release(media);
	libvlc_video_set_callbacks(mp, lock, unlock, display, this);
	libvlc_video_set_format(mp, "RV24", 1920, 1080, 1920 * 3);
	log(Status, "Source setup correctly");
	int height = libvlc_video_get_height(mp);
	int width = libvlc_video_get_width(mp);


}

cv::cuda::GpuMat vlcCamera::doProcess(cv::cuda::GpuMat& img, cv::cuda::Stream& stream)
{
	if (mp)
	{
		libvlc_media_player_play(mp); 
		float fps = libvlc_media_player_get_fps(mp);
		updateParameter("fps", fps);
		cv::cuda::HostMem* h_img;
		imgQueue.wait_and_pop(h_img);
		if (!h_img->empty())
		{
			img.upload(*h_img, stream);
		}		
	}
	return img;
}

bool vlcCamera::SkipEmpty() const
{
	return false;
}


NODE_DEFAULT_CONSTRUCTOR_IMPL(vlcCamera);