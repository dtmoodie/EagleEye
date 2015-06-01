#include "nodes/Utility/DataHandling.h"

using namespace EagleLib;

void GetOutputImage::Init(bool firstInit)
{
    if(firstInit)
        addInputParameter<cv::cuda::GpuMat>("Input");
}

cv::cuda::GpuMat GetOutputImage::doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream)
{
    cv::cuda::GpuMat* input = getParameter<cv::cuda::GpuMat*>("Input")->data;
    if(input == nullptr)
    {
        log(Status, "Input not defined");
        return img;
    }
    if(input->empty())
        log(Status, "Input is empty");
    return *input;
}
cv::cuda::GpuMat ExportInputImage::doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream)
{
    updateParameter("Output image", img, Parameter::Output);
    return img;
}

void ExportInputImage::Init(bool firstInit)
{

}

void ImageInfo::Init(bool firstInit)
{
    EnumParameter dataType;
    dataType.addEnum(ENUM(CV_8U));
    dataType.addEnum(ENUM(CV_8S));
    dataType.addEnum(ENUM(CV_16U));
    dataType.addEnum(ENUM(CV_16S));
    dataType.addEnum(ENUM(CV_32S));
    dataType.addEnum(ENUM(CV_32F));
    dataType.addEnum(ENUM(CV_64F));
    updateParameter<EnumParameter>("Type",dataType, Parameter::State);
}
cv::cuda::GpuMat ImageInfo::doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream)
{

    auto param = getParameter<EnumParameter>(0);
    if(param->data.currentSelection != img.type())
    {
        param->data.currentSelection = img.type();
        parameters[0]->changed = true;
        parameters[0]->onUpdate();
    }
    updateParameter<int>("Depth",img.depth(), Parameter::State);
    updateParameter<int>("Rows",img.rows, Parameter::State);
    updateParameter<int>("Cols", img.cols, Parameter::State);
    updateParameter<int>("Channels", img.channels(), Parameter::State);
    updateParameter<int>("Step", img.step, Parameter::State);
    updateParameter<int>("Ref count", *img.refcount, Parameter::State);
    return img;
}
void Mat2Tensor::Init(bool firstInit)
{
    EnumParameter dataType;
    dataType.addEnum(ENUM(CV_8U));
    dataType.addEnum(ENUM(CV_8S));
    dataType.addEnum(ENUM(CV_16U));
    dataType.addEnum(ENUM(CV_16S));
    dataType.addEnum(ENUM(CV_32S));
    dataType.addEnum(ENUM(CV_32F));
    dataType.addEnum(ENUM(CV_64F));
    updateParameter<EnumParameter>("Tensor Type",dataType);
    updateParameter("Include Position", true);
}
cv::cuda::GpuMat Mat2Tensor::doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream)
{
    int type = getParameter<EnumParameter>(0)->data.currentSelection;
    bool position = getParameter<bool>(1)->data;
    int newCols = img.channels();
    if(position)
        newCols += 2;
    int rows = img.size().area();
    TIME
    if(position && positionMat.empty() || parameters[0]->changed)
    {
        cv::Mat h_positionMat(img.size().area(), 2, type);
        int row = 0;
        for(int y = 0; y < img.rows; ++y)
        {
            for(int x = 0; x < img.cols; ++x, ++row)
            {
                if(type == CV_8U)
                {
                    h_positionMat.at<uchar>(row,0) = x;
                    h_positionMat.at<uchar>(row,1) = y;
                }
                if(type == CV_8S)
                {
                    h_positionMat.at<char>(row,0) = x;
                    h_positionMat.at<char>(row,1) = y;
                }
                if(type == CV_16U)
                {
                    h_positionMat.at<unsigned short>(row,0) = x;
                    h_positionMat.at<unsigned short>(row,1) = y;
                }
                if(type == CV_32S)
                {
                    h_positionMat.at<int>(row,0) = x;
                    h_positionMat.at<int>(row,1) = y;
                }
                if(type == CV_32F)
                {
                    h_positionMat.at<float>(row,0) = x;
                    h_positionMat.at<float>(row,1) = y;
                }
                if(type == CV_64F)
                {
                    h_positionMat.at<double>(row,0) = x;
                    h_positionMat.at<double>(row,1) = y;
                }
            }
        }
        positionMat.upload(h_positionMat, stream);
        parameters[0]->changed = false;
    }
    TIME
    auto buf = bufferPool.getFront();
    auto typeBuf = bufferPool.getFront();
    TIME
    if(position && !positionMat.empty())
    {
        TIME
        buf->data.create(rows, newCols, type);
        TIME
        img.convertTo(typeBuf->data, type,stream);
        TIME
        typeBuf->data.reshape(1, rows).copyTo(buf->data(cv::Rect(0,0,img.channels(),rows)),stream);
        TIME
        positionMat.copyTo(buf->data(cv::Rect(img.channels(),0,2,rows)), stream);
        TIME
        return buf->data;
    }else
    {
        img.convertTo(typeBuf->data, type, stream);
        TIME
        return typeBuf->data.reshape(1, rows);
    }
    return img;
}

NODE_DEFAULT_CONSTRUCTOR_IMPL(GetOutputImage)
NODE_DEFAULT_CONSTRUCTOR_IMPL(ImageInfo)
NODE_DEFAULT_CONSTRUCTOR_IMPL(ExportInputImage)
NODE_DEFAULT_CONSTRUCTOR_IMPL(Mat2Tensor)
