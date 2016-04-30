#include "nodes/IO/ImageLoader.h"
#include <EagleLib/rcc/external_includes/cv_imgcodec.hpp>
#include <boost/filesystem.hpp>
#include <parameters/ParameteredObjectImpl.hpp>
using namespace EagleLib;
using namespace EagleLib::Nodes;


void ImageLoader::Init(bool firstInit)
{
    Node::Init(firstInit);
    updateParameter<Parameters::ReadFile>("Filename", Parameters::ReadFile("/home/dmoodie/Downloads/oimg.jpeg"))->SetTooltip("Path to image file");
    _parameters[0]->changed = true;
}
void ImageLoader::load()
{
	auto path = getParameter<Parameters::ReadFile>(0);
    if(path)
    {
		if (boost::filesystem::exists(*path->Data()))
        {
			cv::Mat h_img = cv::imread(path->Data()->string());
			if (!h_img.empty())
			{
				d_img.upload(h_img);
			}
			else
			{
				NODE_LOG(error) << "Loaded empty image";
			}
        }else
        {
            //log(Status, "File doesn't exist");
			NODE_LOG(warning) << "File doesn't exist";
        }
    }
}

cv::cuda::GpuMat ImageLoader::doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream)
{
    TIME
    if(_parameters[0]->changed)
    {
        load();
        _parameters[0]->changed = false;
        updateParameter("Loaded image", d_img, &stream);
    }
    TIME
	
    return img;
}

bool ImageLoader::SkipEmpty() const
{
    return false;
}
bool DirectoryLoader::SkipEmpty() const
{
    return false;
}
void DirectoryLoader::Init(bool firstInit)
{
    updateParameter<Parameters::ReadDirectory>("Directory", boost::filesystem::path("/home/dan/build/EagleEye/bin"))->SetTooltip( "Path to directory");
    updateParameter<bool>("Repeat", true);
    updateParameter<boost::function<void(void)>>("Restart", boost::bind(&DirectoryLoader::restart, this));
    fileIdx = 0;
}

void DirectoryLoader::restart()
{
    fileIdx = 0;
}

cv::cuda::GpuMat DirectoryLoader::doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream)
{
    if(_parameters[0]->changed)
    {
		boost::filesystem::path& path = *getParameter<Parameters::ReadDirectory>(0)->Data();
        files.clear();
        if(boost::filesystem::is_directory(path))
        {
            for(boost::filesystem::directory_iterator itr(path); itr != boost::filesystem::directory_iterator(); ++itr)
            {
                if(boost::filesystem::is_regular_file(itr->status()))
                    files.push_back(itr->path().string());
            }
        }
    }
    if(files.size())
    {
        if(fileIdx < files.size())
        {
            cv::Mat h_img = cv::imread(files[fileIdx]);
            updateParameter("Current file", files[fileIdx]);
			if (!h_img.empty())
			{
				img.upload(h_img, stream);
				++fileIdx;
			}				
        }
		
			if (fileIdx == files.size())
			{
				//log(Status, "End of directory reached");
				NODE_LOG(info) << "End of directory reached";
				if (*getParameter<bool>(1)->Data())
					fileIdx = 0;

			}
                
    }
    return img;
}



NODE_DEFAULT_CONSTRUCTOR_IMPL(ImageLoader, Image, Source)
NODE_DEFAULT_CONSTRUCTOR_IMPL(DirectoryLoader, Image, Source)

