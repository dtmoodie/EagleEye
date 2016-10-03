#pragma once
#include <boost/function.hpp>
#include <MetaObject/Parameters/ITypedParameter.hpp>


namespace EagleLib
{
    // Qualifiers used for filtering potential inputs
    template<typename T> bool qualifier(mo::IParameter* param, int width = -1, int height = -1, int channels = -1, int type = -1)
    {
        //auto typedParam = getParameterPtr<T>(param);
        auto typedParam_ = dynamic_cast<mo::ITypedParameter<T>*>(param);
        if (typedParam_)
        {
            auto typedParam = typedParam_->Data();
            if (typedParam)
            {
                if (typedParam->channels() == channels || channels == -1)
                {
                    if (typedParam->rows == height || height == -1)
                    {
                        if (typedParam->cols == width || width == -1)
                        {
                            if (typedParam->type() == type || type == -1)
                            {
                                return true;
                            }
                        }
                    }
                }
            }
        }        
        return false;
    }

    template<typename T> class MatQualifier
    {
    public:
        static boost::function<bool(mo::IParameter*)> get(int width = -1, int height = -1, int channels = -1, int type = -1)
        {
            return boost::bind(qualifier<T>,_1,width,height,channels, type);
        }
    };

    typedef MatQualifier<cv::cuda::GpuMat> GpuMatQualifier;
    typedef MatQualifier<cv::Mat> CpuMatQualifier;
}
