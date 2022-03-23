#pragma once
#include "aqcore/imgproc/snakes.hpp"
#include <aqbio/aqbio_export.hpp>
#include <ct/reflect.hpp>

#if __cplusplus <= 201103
#ifndef _MSC_VER
#error "This file requires C++14"
#endif
#endif

namespace aqbio
{

    aqbio_EXPORT void findMaximalSeam(const cv::Mat& score, cv::Mat& seam, int search_window = 5);

    struct Cell
    {
        REFLECT_INTERNAL_BEGIN(Cell)
        REFLECT_INTERNAL_MEMBER(Eigen::Vector2f, center)
        REFLECT_INTERNAL_MEMBER(std::vector<cv::Point>, inner_membrane)
        REFLECT_INTERNAL_MEMBER(std::vector<cv::Point>, outer_membrane)
        REFLECT_INTERNAL_MEMBER(std::vector<float>, inner_point_position_confidence)
        REFLECT_INTERNAL_MEMBER(std::vector<float>, outer_point_position_confidence)
        REFLECT_INTERNAL_MEMBER(bool, inner_updated, false)
        REFLECT_INTERNAL_MEMBER(bool, outer_updated, false)
        REFLECT_INTERNAL_MEMBER(size_t, fn)
REFLECT_INTERNAL_END
;

float dist(const cv::Point& pt) const;

void pushInner(const ssize_t idx, const float dist);

void pushOuter(const ssize_t idx, const float dist);

void clear();

private:
void push(cv::Point& pt, const float dist);
}
;

class aqbio_EXPORT FindCellMembrane : public aqcore::SnakeCircle
{
  public:
    enum Method
    {
        Naive,
        DynamicProgramming
    };

    MO_DERIVE(FindCellMembrane, aqcore::SnakeCircle)
        PARAM(float, confidence_threshold, 0.8f)
        PARAM(float, inner_pad, 1.1f)
        PARAM(float, outer_pad, 1.8f)
        PARAM(float, outer_threshold, 0.1f)
        PARAM(float, radial_resolution, 0.5f)
        PARAM(float, radial_weight, 2.0f)
        PARAM(float, mouse_sigma, 2.0F)
        MO_SLOT(void, mouseDrag, std::string, cv::Point, cv::Point, int, cv::Mat)
        ENUM_PARAM(method, DynamicProgramming, Naive)
        PARAM(bool, user_update, false)
        OUTPUT(Cell, cell, {})
    MO_END;

  protected:
    bool processImpl() override;

    bool snakePoints(const cv::Mat& img, std::vector<cv::Point>& points, const std::vector<float>& point_weight);

    std::vector<cv::Point>
    findOuterMembrane(const cv::Mat& img, const std::vector<cv::Point>& inner, const aq::Circle<float>& circle);

    std::vector<cv::Point>
    findOuterMembraneDP(const cv::Mat& img, const std::vector<cv::Point>& inner, const aq::Circle<float>& circle);
    void reweightEnergy(cv::Mat_<float>& energy);
    Cell m_current_cell;
};
}
