#include "nodes/Node.h"


namespace EagleLib
{
    class Sobel: public Node
    {
    public:
        Sobel();
        virtual void Init(bool firstInit);
        virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream);

    };

    class Canny: public Node
    {
    public:
        Canny();
        virtual void Init(bool firstInit);
        virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream);
    };

    class Laplacian: public Node
    {
    public:
        Laplacian();
        virtual void Init(bool firstInit);
        virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream);
    };
    class BiLateral: public Node
    {
    public:
        BiLateral();
        virtual void Init(bool firstInit);
        virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream);
    };
    class MeanShiftFilter: public Node
    {
    public:
        MeanShiftFilter();
        virtual void Init(bool firstInit);
        virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream);
    };
    class MeanShiftProc: public Node
    {
    public:
        MeanShiftProc();
        virtual void Init(bool firstInit);
        virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream);
    };
    class MeanShiftSegmentation: public Node
    {
    public:
        MeanShiftSegmentation();
        virtual void Init(bool firstInit);
        virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream);
    };
}
