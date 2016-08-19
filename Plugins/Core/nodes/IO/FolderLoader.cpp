#include "FolderLoader.h"
#include <boost/filesystem.hpp>
#include "EagleLib/Detail/Export.hpp"
#include <parameters/ParameteredObjectImpl.hpp>
using namespace EagleLib;
using namespace EagleLib::Nodes;

void FolderLoader::backgroundThread(boost::filesystem::path path)
{
    int count = 0;
    boost::filesystem::directory_iterator end_itr;
    cv::cuda::Stream stream;
    while ((count == 0 || *getParameter<bool>(1)->Data()) && !boost::this_thread::interruption_requested())
    {
        for (boost::filesystem::directory_iterator itr(path); itr != end_itr; ++itr)
        {
            if (boost::filesystem::is_regular_file(itr->path()))
            {
                cv::Mat img = cv::imread(itr->path().string());

                if (!img.empty())
                {
                    cv::cuda::registerPageLocked(img);
                    cv::cuda::GpuMat d_img;
                    d_img.upload(img, stream);
                    stream.waitForCompletion();
                    cv::cuda::unregisterPageLocked(img);
                    imageNotifier.wait_push(d_img);
                }
            }
        }
        ++count;
    }
}

void FolderLoader::onDirectoryChange()
{
    thread.interrupt();
    thread.join();
    thread = boost::thread(boost::bind(&FolderLoader::backgroundThread, this, 
        boost::filesystem::path(getParameter<Parameters::ReadDirectory>(0)->Data()->string())));
}

void FolderLoader::NodeInit(bool firstInit)
{
    updateParameter<Parameters::ReadDirectory>("Load Directory", boost::filesystem::path(""));
    updateParameter<bool>("Loop", true);
    RegisterParameterCallback(0, boost::bind(&FolderLoader::onDirectoryChange, this));
}

cv::cuda::GpuMat FolderLoader::doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream)
{
    imageNotifier.wait_and_pop(img);
    return img;
}

NODE_DEFAULT_CONSTRUCTOR_IMPL(FolderLoader, Image, Source);