#include <aqcore_export.hpp>

#include <Aquila/nodes/Node.hpp>
#include <Aquila/types/Stamped.hpp>
#include <Aquila/types/SyncedMemory.hpp>

#include <opencv2/imgproc.hpp>

#include <RuntimeObjectSystem/RuntimeInclude.h>
#include <RuntimeObjectSystem/RuntimeSourceDependency.h>

RUNTIME_COMPILER_SOURCEDEPENDENCY
RUNTIME_MODIFIABLE_INCLUDE

namespace cv
{
    namespace cuda
    {
        class Filter;
    }
} // namespace cv

namespace aq
{
    namespace nodes
    {

        class MorphologyFilter : public Node
        {
          public:
            MO_DERIVE(MorphologyFilter, Node)
                INPUT(SyncedImage, input_image)
                OUTPUT(SyncedImage, output)
                ENUM_PARAM(structuring_element_type, cv::MORPH_RECT, cv::MORPH_CROSS, cv::MORPH_ELLIPSE)
                ENUM_PARAM(morphology_type,
                           cv::MORPH_ERODE,
                           cv::MORPH_DILATE,
                           cv::MORPH_OPEN,
                           cv::MORPH_CLOSE,
                           cv::MORPH_GRADIENT,
                           cv::MORPH_TOPHAT,
                           cv::MORPH_BLACKHAT)
                PARAM(int, iterations, 1)
                PARAM(cv::Mat, structuring_element, cv::getStructuringElement(0, cv::Size(5, 5)))
                PARAM(cv::Point, anchor_point, cv::Point(-1, -1))
                PARAM(int, structuring_element_size, 5)
            MO_END

          protected:
            virtual bool processImpl() override;
            cv::Ptr<::cv::cuda::Filter> filter;
        };

        class FindContours : public Node
        {
          public:
            MO_DERIVE(FindContours, Node)
                INPUT(SyncedImage, input_image)
                ENUM_PARAM(mode, cv::RETR_EXTERNAL, cv::RETR_LIST, cv::RETR_CCOMP, cv::RETR_TREE, cv::RETR_FLOODFILL)
                ENUM_PARAM(method,
                           cv::CHAIN_APPROX_NONE,
                           cv::CHAIN_APPROX_SIMPLE,
                           cv::CHAIN_APPROX_TC89_L1,
                           cv::CHAIN_APPROX_TC89_KCOS)

                PARAM(bool, calculate_contour_area, false)
                PARAM(bool, calculate_moments, false)
                STATUS(int, num_contours, 0)

                OUTPUT(std::vector<std::vector<cv::Point>>, contours, {})
                OUTPUT(std::vector<cv::Vec4i>, hierarchy, {})
            MO_END

          protected:
            virtual bool processImpl() override;
        };

        class PruneContours : public Node
        {
          public:
            MO_DERIVE(PruneContours, Node)
                PARAM(int, min_area, 20)
                PARAM(int, max_area, 500)
            MO_END
            PruneContours();

            virtual void nodeInit(bool firstInit) override;
            virtual TS<SyncedMemory> doProcess(TS<SyncedMemory> img, cv::cuda::Stream& stream);
        };

        class ContourBoundingBox : public Node
        {
          public:
            typedef std::vector<std::pair<int, double>> contour_area_t;
            MO_DERIVE(ContourBoundingBox, Node)
                INPUT(SyncedImage, input_image)
                INPUT(std::vector<std::vector<cv::Point>>, contours)
                INPUT(std::vector<cv::Vec4i>, hierarchy)
                PARAM(cv::Scalar, box_color, (cv::Scalar(0, 0, 255)))
                PARAM(int, line_thickness, 2)
                PARAM(bool, use_filtered_area, false)
                PARAM(bool, merge_contours, false)
                PARAM(int, separation_distance, false)
                OUTPUT(contour_area_t, contour_area)
            MO_END
          protected:
            virtual bool processImpl() override;
            ContourBoundingBox();
        };

        class HistogramThreshold : public Node
        {
            cv::cuda::GpuMat* inputHistogram;
            cv::cuda::GpuMat* inputImage;
            cv::cuda::GpuMat* inputMask;
            cv::cuda::Stream _stream;
            cv::cuda::GpuMat lowerMask;
            cv::cuda::GpuMat upperMask;
            enum ThresholdType
            {
                KeepCenter = 0,
                SuppressCenter
            };
            ThresholdType type;

          public:
            HistogramThreshold();
            virtual void nodeInit(bool firstInit);
            virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat& img, cv::cuda::Stream& stream);
            void runFilter();
        };

        class DrawContours : public Node
        {
          public:
            enum DrawMode
            {
                LargestSize,
                LargestArea,
                All
            };

            MO_DERIVE(DrawContours, Node)
                INPUT(SyncedImage, input_image)
                INPUT(std::vector<std::vector<cv::Point>>, input_contours)
                PARAM(cv::Scalar, draw_color, cv::Scalar(0, 0, 255))
                PARAM(int, draw_thickness, 8)
                ENUM_PARAM(draw_mode, LargestSize, LargestArea, All)
                OUTPUT(SyncedImage, output)
            MO_END
          protected:
            virtual bool processImpl() override;
        };

        class DrawRects : public Node
        {
          public:
            DrawRects();
            virtual void nodeInit(bool firstInit);
            virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat& img, cv::cuda::Stream& stream);
        };

    } // namespace nodes
} // namespace aq
