#pragma once

#include "nodes/Node.h"


namespace EagleLib
{
    struct ColorScale
    {
        ColorScale(double start_ = 0, double slope_ = 1, bool symmetric_ = false);
        // Defines where this color starts to take effect, between zero and 1000
        double start;
        // Defines the slope of increase / decrease for this color between 1 and 255
        double slope;
        // Defines if the slope decreases after it peaks
        bool	symmetric;
        // Defines if this color starts high then goes low
        bool	inverted;
        bool flipped;
        uchar operator ()(double location);
        uchar getValue(double location_);

    };
    class AutoScale: public Node
    {
    public:
        AutoScale();
        virtual void Init(bool firstInit);
        virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat &img);
    };

    class Colormap: public Node
    {
        ColorScale red, green, blue;
        std::vector<cv::Vec3b> LUT;
        int resolution;
        double scale, shift;
        void buildLUT();
    public:
        Colormap();
        virtual void Init(bool firstInit);
        virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat &img);
    };
    class Normalize: public Node
    {
    public:
        Normalize();
        virtual void Init(bool firstInit);
        virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat &img);
    };



}
