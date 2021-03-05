#pragma once

#include <Aquila/types/ObjectDetection.hpp>
#include <Aquila/types/SyncedImage.hpp>
#include <Aquila/types/geometry/Circle.hpp>
#include <Aquila/types/geometry/Contour.hpp>

#include <opencv2/core.hpp>

#include <Aquila/nodes/Node.hpp>

namespace aqcore
{
    AQUILA_EXPORTS bool snakePoints(const cv::Mat& img,
                                    std::vector<cv::Point>& points,
                                    int kernel_size,
                                    cv::TermCriteria term_crit,
                                    int mode,
                                    float* alpha,
                                    float* beta,
                                    float* gamma,
                                    int coefficients = 1,
                                    const float* position_weight = nullptr);

    AQUILA_EXPORTS void sampleCircle(std::vector<cv::Point>& pts, aq::Circle<float> circle, float pad, int samples);

    class AQUILA_EXPORTS SnakeCircle : public aq::nodes::Node
    {
      public:
        enum SnakeMode
        {
            GRAD = 2,
            IMAGE = 1
        };
        using Components_t = ct::VariadicTypedef<aq::Circlef>;
        using Input_t = aq::TEntityComponentSystem<Components_t>;

        MO_DERIVE(SnakeCircle, aq::nodes::Node)
            INPUT(aq::SyncedImage, input)
            INPUT(Input_t, circles)

            PARAM(float, alpha, 0.1f)
            PARAM(float, beta, 0.4f)
            PARAM(float, gamma, 0.5f)
            PARAM(float, pad, 1.0F)
            PARAM(int, kernel_size, 3)
            PARAM(int, window_size, 9)
            PARAM(int, iterations, 1000)
            PARAM(int, num_samples, 200)
            ENUM_PARAM(mode, GRAD, IMAGE)

            OUTPUT(aq::TEntityComponentSystem<ct::VariadicTypedef<aq::Contour>>, output)
        MO_END;

      protected:
        bool snakePoints(const cv::Mat& img, std::vector<cv::Point>& points);
        bool processImpl() override;
    };

} // namespace aqcore
