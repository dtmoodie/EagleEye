#pragma once
#include "parameterplotter.h"
#include <boost/circular_buffer.hpp>
// Includes all the tempalates for plotting different datatypes

class ParameterPlotter;

class ParameterPlotterFactory
{
public:
    virtual ParameterPlotter* create(EagleLib::Parameter::Ptr param, QCustomPlot* plot_ = nullptr) = 0;
    virtual bool acceptsType(EagleLib::Parameter::Ptr param) = 0;
    virtual QString plotName() = 0;
};

static QList<ParameterPlotterFactory*> g_factories;
cv::Size inline getSize(EagleLib::Parameter::Ptr param)
{
    auto gpuParam = EagleLib::getParameterPtr<cv::cuda::GpuMat>(param);
    if(gpuParam)
        return gpuParam->size();
    auto cpuParam = EagleLib::getParameterPtr<cv::Mat>(param);
    if(cpuParam)
        return cpuParam->size();
    auto hostParam = EagleLib::getParameterPtr<cv::cuda::HostMem>(param);
    if(hostParam)
        return hostParam->size();

    return cv::Size(0,0);
}
int inline getChannels(EagleLib::Parameter::Ptr param)
{
    auto gpuParam = EagleLib::getParameterPtr<cv::cuda::GpuMat>(param);
    if(gpuParam)
        return gpuParam->channels();
    auto cpuParam = EagleLib::getParameterPtr<cv::Mat>(param);
    if(cpuParam)
        return cpuParam->channels();
    auto hostParam = EagleLib::getParameterPtr<cv::cuda::HostMem>(param);
    if(hostParam)
        return hostParam->channels();
    return 0;
}
QVector<double> inline getParamArrayDataHelper(cv::Mat h_data, int channel)
{
    QVector<double> output;
    CV_Assert((h_data.rows == 1 || h_data.cols == 1) && channel < h_data.channels());
    const int numItems = h_data.size().area();
    output.reserve(numItems);

    for(int i = 0; i < numItems; ++i)
    {
        if(h_data.depth() == CV_8U)
            output.push_back(h_data.at<uchar>(i));
        if(h_data.depth() == CV_8UC3)
            output.push_back(h_data.at<cv::Vec2b>(i).val[channel]);
        if(h_data.depth() == CV_16U)
            output.push_back(h_data.at<unsigned short>(i));
        if(h_data.depth() == CV_16UC3)
            output.push_back(h_data.at<cv::Vec3w>(i).val[channel]);
        if(h_data.depth() == CV_16S)
            output.push_back(h_data.at<short>(i));
        if(h_data.depth() == CV_16SC3)
            output.push_back(h_data.at<cv::Vec3s>(i).val[channel]);
        if(h_data.depth() == CV_32S)
            output.push_back(h_data.at<int>(i));
        if(h_data.depth() == CV_32SC3)
            output.push_back(h_data.at<cv::Vec3i>(i).val[channel]);
        if(h_data.depth() == CV_32F)
            output.push_back(h_data.at<float>(i));
        if(h_data.depth() == CV_32FC3)
            output.push_back(h_data.at<cv::Vec3f>(i).val[channel]);
        if(h_data.depth() == CV_64F)
            output.push_back(h_data.at<double>(i));
        if(h_data.depth() == CV_64FC3)
            output.push_back(h_data.at<cv::Vec3d>(i).val[channel]);
    }
    return output;
}

QVector<double> inline getParamArrayData(EagleLib::Parameter::Ptr param, int channel)
{
    auto gpuParam = EagleLib::getParameterPtr<cv::cuda::GpuMat>(param);
    if(gpuParam)
        return getParamArrayDataHelper(cv::Mat(*gpuParam), channel);
    auto cpuParam = EagleLib::getParameterPtr<cv::Mat>(param);
    if(cpuParam)
        return getParamArrayDataHelper(*cpuParam, channel);
    auto hostParam = EagleLib::getParameterPtr<cv::cuda::HostMem>(param);
    if(hostParam)
        return getParamArrayDataHelper(hostParam->createMatHeader(), channel);
    auto vecParam = EagleLib::getParameterPtr<std::vector<double>>(param);
    if(vecParam)
        return getParamArrayDataHelper(cv::Mat(*vecParam), 0);
    return QVector<double>();
}
template<typename T> bool inline getData(EagleLib::Parameter::Ptr param, double& data)
{
    auto ptr = EagleLib::getParameterPtr<T>(param);
    if(ptr)
    {
        data = *ptr;
        return true;
    }
    return false;
}
template<typename T> bool inline getDataVec(EagleLib::Parameter::Ptr param, int channel, double& data)
{
    auto ptr = EagleLib::getParameterPtr<T>(param);
    if(ptr)
    {
        data = ptr->val[channel];
        return true;
    }
    return false;
}

double inline getParamData(EagleLib::Parameter::Ptr data, int channel)
{
    double output;
    if(getData<double>(data, output))
        return output;
    if(getData<int>(data,output))
        return output;
    if(getData<char>(data,output))
        return output;
    if(getData<unsigned char>(data,output))
        return output;
    if(getData<float>(data,output))
        return output;
    if(getData<unsigned int>(data,output))
        return output;
    if(getData<short>(data,output))
        return output;
    if(getData<unsigned short>(data,output))
        return output;
    if(getDataVec<cv::Vec2b>(data, channel, output))
        return output;
    if(getDataVec<cv::Vec3b>(data, channel, output))
        return output;
    if(getDataVec<cv::Vec4b>(data, channel, output))
        return output;
    if(getDataVec<cv::Vec2s>(data, channel, output))
        return output;
    if(getDataVec<cv::Vec3s>(data, channel, output))
        return output;
    if(getDataVec<cv::Vec4s>(data, channel, output))
        return output;
    if(getDataVec<cv::Vec2i>(data, channel, output))
        return output;
    if(getDataVec<cv::Vec3i>(data, channel, output))
        return output;
    if(getDataVec<cv::Vec4i>(data, channel, output))
        return output;
    if(getDataVec<cv::Vec6i>(data, channel, output))
        return output;
    if(getDataVec<cv::Vec8i>(data, channel, output))
        return output;
    if(getDataVec<cv::Vec2d>(data, channel, output))
        return output;
    if(getDataVec<cv::Vec3d>(data, channel, output))
        return output;
    if(getDataVec<cv::Vec4d>(data, channel, output))
        return output;
    if(getDataVec<cv::Vec6d>(data, channel, output))
        return output;
    if(getDataVec<cv::Scalar>(data, channel, output))
        return output;
}


// Helper policies
struct DefaultChannelPolicy
{
    static bool acceptsChannels(int channels){return channels != 0;}
};

struct SingleChannelPolicy
{
    static bool acceptsChannels(int channels){ return channels == 1;}
};

struct MultiChannelPolicy
{
    static bool acceptsChannels(int channels){ return channels > 1;}
};

struct DefaultSizePolicy
{
    static bool acceptsSize(cv::Size size){ return true;}
};

struct VectorSizePolicy
{
    static bool acceptsSize(cv::Size size){ return size.width == 1 || size.height == 1;}
};

struct MatrixSizePolicy
{
    static bool acceptsSize(cv::Size size){ return size.width != 1 && size.height != 1;}
};

struct DefaultTypePolicy
{
    static bool acceptsType(EagleLib::Parameter::Ptr param)
    {
        return true;
    }
};

template<typename T> struct TypePolicy
{
    static bool acceptsType(EagleLib::Parameter::Ptr param)
    {
        return EagleLib::acceptsType<T>(param->typeInfo);
    }
};
struct StaticPlotPolicy
{
    int size;
    int channel;
    QVector<double> data;
    void addPlotData(EagleLib::Parameter::Ptr param)
    {
        data = getParamArrayData(param, channel);
    }

    QVector<double> getPlotData()
    {
        return data;
    }

    void setSize(int size_)
    {
        size = size_;
    }
};

struct SlidingWindowPlotPolicy
{
    int channel;

    boost::circular_buffer<double> plotData;
    void addPlotData(EagleLib::Parameter::Ptr param)
    {
        plotData.push_back(getParamData(param, channel));
    }

    QVector<double> getPlotData()
    {
        QVector<double> data;
        data.reserve(plotData.size());
        for(int i = 0; i < plotData.size(); ++i)
        {
            data.push_back(plotData[i]);
        }
        return data;
    }

    void setSize(int size)
    {
        plotData.set_capacity(size);
    }
};


template<typename Plotter, typename ChannelPolicy_, typename SizePolicy_,
         typename TypePolicy1, typename TypePolicy2 = DefaultTypePolicy, typename TypePolicy3 = DefaultTypePolicy,
         typename TypePolicy4 = DefaultTypePolicy, typename TypePolicy5 = DefaultTypePolicy,
         typename TypePolicy6 = DefaultTypePolicy, typename TypePolicy7 = DefaultTypePolicy
         > struct PlotterFactory: public ParameterPlotterFactory
{
    const char* name;
    PlotterFactory(const char* name_):
        name(name_)
    {
        g_factories.push_back(this);
    }
    virtual ParameterPlotter* create(EagleLib::Parameter::Ptr param, QCustomPlot* plot_ = nullptr)
    {
        return new Plotter(param, plot_);
    }
    virtual bool acceptsType(EagleLib::Parameter::Ptr param)
    {
        bool b1 = SizePolicy_::acceptsSize(getSize(param));
        bool b2 = ChannelPolicy_::acceptsChannels(getChannels(param));
        bool b3 = Plotter::acceptsType(param);
        return b1 && b2 && b3;
//        bool b3 = TypePolicy1::acceptsType(param);
//        bool b4 = TypePolicy2::acceptsType(param);
//        bool b5 = TypePolicy3::acceptsType(param);
//        bool b6 = TypePolicy4::acceptsType(param);
//        bool b7 = TypePolicy5::acceptsType(param);
//        bool b8 = TypePolicy6::acceptsType(param);
//        bool b9 = TypePolicy7::acceptsType(param);

//        return b1 && b2 && (b3 || b4 || b5 || b6 || b7 || b8 || b9);
    }
    virtual QString plotName()
    {
        return QString(name);
    }
};


