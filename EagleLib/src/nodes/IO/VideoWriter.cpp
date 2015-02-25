#include <nodes/IO/VideoWriter.h>
using namespace EagleLib;
using namespace EagleLib::IO;

VideoWriter::VideoWriter()
{
	addParameter(std::string("WriteImgFunction"), boost::bind(&VideoWriter::writeImg, this, _1), "Function for writing image to a video file", Parameter::Output);
	addParameter(std::string("FileName"), std::string(), "Absolute path to file being writen");
	addParameter(std::string("FourCC"), "MJPG", "Codec to use for writing videos");
	try
	{
		cv::cudacodec::createVideoWriter("test.avi", cv::Size(640, 480), 30);
	}
	catch (cv::Exception &e)
	{
		e.code;
		gpuWriter = false;
	}
}

VideoWriter::VideoWriter(std::string fileName)
{
	updateParameter(1, fileName);
}

VideoWriter::~VideoWriter()
{

}

cv::cuda::GpuMat 
VideoWriter::doProcess(cv::cuda::GpuMat& img)
{
	return img;
}
void 
VideoWriter::writeImg(cv::cuda::GpuMat& img)
{
	auto fileName = getParameter<std::string>(1);
	if (fileName->data.size() == 0)
		return;
	static cv::Size imgSize = img.size();
	if (imgSize != img.size())
	{

	}
	if (gpuWriter)
	{
		if (d_writer == NULL)
			d_writer = cv::cudacodec::createVideoWriter(fileName->data, imgSize, 30.0);
		//d_writer->write(img);
		return;
	}
	if (!gpuWriter)
	{
		if (!h_writer.isOpened())
		{
			auto fourcc = getParameter<const char*>(2);
			h_writer.open(fileName->data, cv::VideoWriter::fourcc(fourcc->data[0], fourcc->data[1], fourcc->data[2], fourcc->data[3]), 30, imgSize, img.channels() == 3);
		}
		cv::Mat h_img(img);
		h_writer << h_img;
		return;
	}	
}

REGISTERCLASS(VideoWriter)