#include <aqcore_export.hpp>
#include <ct/types/opencv.hpp>

#include "../OpenCVCudaNode.hpp"

#include <Aquila/nodes/Node.hpp>
#include <Aquila/types/SyncedImage.hpp>

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

namespace aqcore
{
    /**
     * @brief The MorphologyFilter class applies standard opencv morphological operations to the input image
     */
    class MorphologyFilter : public OpenCVCudaNode
    {
      public:
        MO_DERIVE(MorphologyFilter, OpenCVCudaNode)
            INPUT(aq::SyncedImage, input)

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

            OUTPUT(aq::SyncedImage, output)
        MO_END;

      protected:
        bool processImpl(aq::CVStream&) override;
        cv::Ptr<::cv::cuda::Filter> filter;
    };

    class FindContours : public OpenCVCudaNode
    {
        // TODO: Update to using ECS
      public:
        MO_DERIVE(FindContours, aq::nodes::Node)
            INPUT(aq::SyncedImage, input)

            ENUM_PARAM(mode, cv::RETR_EXTERNAL, cv::RETR_LIST, cv::RETR_CCOMP, cv::RETR_TREE, cv::RETR_FLOODFILL)
            ENUM_PARAM(method,
                       cv::CHAIN_APPROX_NONE,
                       cv::CHAIN_APPROX_SIMPLE,
                       cv::CHAIN_APPROX_TC89_L1,
                       cv::CHAIN_APPROX_TC89_KCOS)

            PARAM(bool, calculate_contour_area, false)
            PARAM(bool, calculate_moments, false)
            STATUS(int, num_contours, 0)

            OUTPUT(std::vector<std::vector<cv::Point>>, contours)
            OUTPUT(std::vector<cv::Vec4i>, hierarchy)
        MO_END;

      protected:
        virtual bool processImpl(mo::IAsyncStream&) override;
    };

    class PruneContours : public aq::nodes::Node
    {
        // TODO: Update to using ECS
      public:
        MO_DERIVE(PruneContours, aq::nodes::Node)
            INPUT(std::vector<std::vector<cv::Point>>, input)
            INPUT(std::vector<cv::Vec4i>, hierarchy)

            PARAM(int, min_area, 20)
            PARAM(int, max_area, 500)

            OUTPUT(std::vector<std::vector<cv::Point>>, output)
        MO_END;
        PruneContours();

      private:
        void nodeInit(bool firstInit) override;
        bool processImpl(mo::IAsyncStream&) override;
    };

    /**
     * @brief The ContourBoundingBox class extracts bounding boxes as cv::Rects from input contours
     *   TODO: Update to using ECS
     */
    class ContourBoundingBox : public aq::nodes::Node
    {
      public:
        MO_DERIVE(ContourBoundingBox, aq::nodes::Node)
            INPUT(std::vector<std::vector<cv::Point>>, contours)
            INPUT(std::vector<cv::Vec4i>, hierarchy)

            PARAM(cv::Scalar, box_color, (cv::Scalar(0, 0, 255)))
            PARAM(int, line_thickness, 2)
            PARAM(bool, use_filtered_area, false)
            PARAM(bool, merge_contours, false)
            PARAM(int, separation_distance, false)

            OUTPUT(std::vector<cv::Rect>, output)
        MO_END;

      protected:
        bool processImpl(mo::IAsyncStream&) override;
        ContourBoundingBox();
    };

    class DrawContours : public aq::nodes::Node
    {
      public:
        enum DrawMode
        {
            LargestSize,
            LargestArea,
            All
        };

        MO_DERIVE(DrawContours, aq::nodes::Node)
            INPUT(aq::SyncedImage, image)
            INPUT(std::vector<std::vector<cv::Point>>, contours)
            PARAM(cv::Scalar, draw_color, cv::Scalar(0, 0, 255))
            PARAM(int, draw_thickness, 8)
            ENUM_PARAM(draw_mode, LargestSize, LargestArea, All)
            OUTPUT(aq::SyncedImage, output)
        MO_END;

      protected:
        bool processImpl(mo::IAsyncStream&) override;
        bool processImpl() override { return false; }
    };

    class DrawRects : public OpenCVCudaNode
    {
      public:
        DrawRects();
        virtual void nodeInit(bool firstInit);
        bool processImpl(aq::CVStream&);
    };

} // namespace aqcore
