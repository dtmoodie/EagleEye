#pragma once
#include <opencv2/core/types.hpp>
#include <opencv2/core/mat.hpp>

#include <vector>

namespace vclick
{
    class BoundingBox
    {
    public:
        inline bool Contains(const cv::Vec3f& point)
        {
            return point[0] > x && point[0] < x + width &&
                point[1] > y && point[1] < y + height &&
                point[2] > z && point[2] < z + depth;
        }
        inline bool Contains(const cv::Point3f& point)
        {
            return point.x > x && point.x < x + width &&
                point.y > y && point.y < y + height &&
                point.z > z && point.z < z + depth;
        }
        cv::Mat Contains(cv::Mat points);
        cv::Mat Contains(std::vector<cv::Vec3f>& points);
        template<typename AR> void serialize(AR& ar);

        float x, y, z;
        float width, height, depth;
    };
}