#include "nodes/IO/ImageLoader.h"
#include <opencv2/imgcodecs.hpp>
using namespace EagleLib;



void ImageLoader::Init(bool firstInit)
{
    updateParameter<boost::filesystem::path>("Filename", boost::filesystem::path("/home/dmoodie/Downloads/oimg.jpeg"), Parameter::Control, "Path to image file");
    parameters[0]->changed = true;
}
void ImageLoader::load()
{
    auto path = getParameter<boost::filesystem::path>(0);
    if(path)
    {
        if(boost::filesystem::exists(path->data))
        {
            cv::Mat h_img = cv::imread(path->data.string()); 
            d_img.upload(h_img); 
        }else
        {
            log(Status, "File doesn't exist");
        }
    }
}

cv::cuda::GpuMat ImageLoader::doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream)
{
    TIME
    if(parameters[0]->changed)
    {
        load();
        parameters[0]->changed = false;
    }
    TIME
	if (!d_img.empty())
    {
        TIME
        d_img.copyTo(img,stream);
        TIME
    }
    return img;
}

bool ImageLoader::SkipEmpty() const
{
    return false;
}
NODE_DEFAULT_CONSTRUCTOR_IMPL(ImageLoader)
