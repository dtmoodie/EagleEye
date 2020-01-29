#include "MeasureCell.hpp"
#include <Aquila/nodes/NodeInfo.hpp>
#include <Eigen/Geometry>
#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics.hpp>
#include <boost/filesystem.hpp>
#include <fstream>
#include <opencv2/imgproc.hpp>

namespace ba = boost::accumulators;
namespace bat = ba::tag;

// https://gamedev.stackexchange.com/questions/44720/line-intersection-from-parametric-equation?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
template <class T>
bool intersects(const Eigen::ParametrizedLine<T, 2>& L,
                T L_len,
                const Eigen::ParametrizedLine<T, 2>& M,
                T M_len,
                Eigen::Matrix<T, 2, 1>* pt = nullptr,
                T* l_intersection_dist = nullptr,
                T* m_intersection_dist = nullptr)
{
    const auto& a = L.origin();
    const auto& b = L.direction();
    const auto& c = M.origin();
    const auto& d = M.direction();
    T u = (c(1) * b(0) - a(1) * b(0) - c(0) * b(1) + a(0) * b(1)) / (d(0) * b(1) - d(1) * b(0));
    T t = (d(0) * (a(1) - c(1)) + d(1) * (c(0) - a(0))) / (b(0) * d(1) - b(1) * d(0));
    if (u >= 0 && u < M_len && t >= 0 && t < L_len)
    {
        if (pt)
        {
            *pt = M.pointAt(u);
        }
        if (l_intersection_dist)
            *l_intersection_dist = t;
        if (m_intersection_dist)
            *m_intersection_dist = u;
        return true;
    }
    return false;
}

namespace aq
{
namespace bio
{
typedef ba::accumulator_set<float, ba::features<bat::mean, bat::max, bat::min, bat::variance>> Accumulator_t;

struct Stats
{
    Stats(const Accumulator_t& acc)
    {
        min = ba::min(acc);
        max = ba::max(acc);
        mean = ba::mean(acc);
        variance = ba::variance(acc);
        stddev = std::sqrt(variance);
    }
    bool passes(float val, float stddev_thresh)
    {
        return (val > (mean - stddev_thresh * stddev) && val < (mean + stddev_thresh * stddev));
    }
    float min, max, mean, variance, stddev;
};

bool MeasureCell::processImpl()
{
    std::vector<Measurement> measurements;
    cv::Point2i centroid(cell->center(0), cell->center(1));
    Accumulator_t inner_radius, outer_radius, diff, inner_diameter, outer_diameter;
    for (size_t i = 0; i < cell->inner_membrane.size(); ++i)
    {
        Measurement measurement;
        cv::Point pt_inner = cell->inner_membrane[i];
        cv::Point pt_outer = cell->outer_membrane[i];
        measurement.outer0 = pt_outer;
        measurement.inner0 = pt_inner;
        float inner = cv::norm(pt_inner - centroid);
        float outer = cv::norm(pt_outer - centroid);
        inner_radius(inner);
        outer_radius(outer);
        diff(outer - inner);
        measurement.inner_radius = inner;
        measurement.outer_radius = outer;
        measurement.diff = outer - inner;
        auto outer_ray_through_center =
            Eigen::ParametrizedLine<float, 2>::Through({pt_outer.x, pt_outer.y}, {centroid.x, centroid.y});
        float d0 = cv::norm(pt_outer - centroid) * 2.5;
        // Find the intersection on the other side
        for (size_t j = 1; j < cell->outer_membrane.size(); ++j)
        {
            if (j != i && j - 1 != i)
            {
                auto p0 = cell->outer_membrane[j - 1];
                auto p1 = cell->outer_membrane[j];
                auto line_segment = Eigen::ParametrizedLine<float, 2>::Through({p0.x, p0.y}, {p1.x, p1.y});
                float d = cv::norm(p0 - p1);
                Eigen::Matrix<float, 2, 1> intersection_point;
                float dist0, dist1;
                if (intersects(outer_ray_through_center, d0, line_segment, d, &intersection_point, &dist0, &dist1))
                {
                    outer_diameter(dist0);
                    measurement.outer_diameter = dist0;
                    measurement.outer1.x = intersection_point(0);
                    measurement.outer1.y = intersection_point(1);
                    break;
                }
            }
        }
        auto inner_ray_through_center =
            Eigen::ParametrizedLine<float, 2>::Through({pt_inner.x, pt_inner.y}, {centroid.x, centroid.y});
        d0 = cv::norm(pt_inner - centroid) * 2.5;
        for (size_t j = 1; j < cell->inner_membrane.size(); ++j)
        {
            if (j != i && j - 1 != i)
            {
                auto p0 = cell->inner_membrane[j - 1];
                auto p1 = cell->inner_membrane[j];
                auto line_segment = Eigen::ParametrizedLine<float, 2>::Through({p0.x, p0.y}, {p1.x, p1.y});
                float d = cv::norm(p0 - p1);
                Eigen::Matrix<float, 2, 1> intersection_point;
                float dist0, dist1;
                if (intersects(inner_ray_through_center, d0, line_segment, d, &intersection_point, &dist0, &dist1))
                {
                    inner_diameter(dist0);
                    measurement.inner_diameter = dist0;
                    measurement.inner1.x = intersection_point(0);
                    measurement.inner1.y = intersection_point(1);
                    break;
                }
            }
        }
        measurements.push_back(std::move(measurement));
    }
    Stats inner_radius_stats(inner_radius);
    Stats outer_radius_stats(outer_radius);
    Stats diff_stats(diff);
    Stats inner_diameter_stats(inner_diameter);
    Stats outer_diameter_stats(outer_diameter);
    cv::Mat draw_image;
    image->clone(draw_image, _ctx.get());
    cv::circle(draw_image, centroid, 5, cv::Scalar(0, 255, 0));
    boost::filesystem::path path;
    if (out_dir.string().empty())
    {
        path = boost::filesystem::path(*image_name);
    }
    else
    {
        if (!boost::filesystem::exists(out_dir))
        {
            boost::filesystem::create_directories(out_dir);
        }
        path = out_dir.string() + "/";
        path += (boost::filesystem::path(*image_name).filename());
    }

    path.replace_extension("");
    std::ofstream ofs(path.string() + "_measurements.csv");
    ofs << "Inner area: " << cv::contourArea(cell->inner_membrane) / (pixel_per_um * pixel_per_um) << '\n';
    ofs << "Outer area: " << cv::contourArea(cell->outer_membrane) / (pixel_per_um * pixel_per_um) << '\n';
    ofs << "Measurement Name, Inner Radius, Outer Radius, Inner Diameter, Outer Diameter, Radius Difference\n";
    ofs << "Mean, " << inner_radius_stats.mean / pixel_per_um << ", " << outer_radius_stats.mean / pixel_per_um << ", "
        << inner_diameter_stats.mean / pixel_per_um << ", " << outer_diameter_stats.mean / pixel_per_um << ", "
        << diff_stats.mean / pixel_per_um << "\n";
    ofs << "Min, " << inner_radius_stats.min / pixel_per_um << ", " << outer_radius_stats.min / pixel_per_um << ", "
        << inner_diameter_stats.min / pixel_per_um << ", " << outer_diameter_stats.min / pixel_per_um << ", "
        << diff_stats.min / pixel_per_um << "\n";
    ofs << "Max, " << inner_radius_stats.max / pixel_per_um << ", " << outer_radius_stats.max / pixel_per_um << ", "
        << inner_diameter_stats.max / pixel_per_um << ", " << outer_diameter_stats.max / pixel_per_um << ", "
        << diff_stats.max / pixel_per_um << "\n";
    ofs << "Stddev, " << inner_radius_stats.stddev / pixel_per_um << ", " << outer_radius_stats.stddev / pixel_per_um
        << ", " << inner_diameter_stats.stddev / pixel_per_um << ", " << outer_diameter_stats.stddev / pixel_per_um
        << ", " << diff_stats.stddev / pixel_per_um << "\n";
    int count = 0;
    for (size_t i = 0; i < measurements.size();)
    {
        const Measurement& measurement = measurements[i];
        if (inner_diameter_stats.passes(measurement.inner_diameter, stddev_thresh) &&
            outer_diameter_stats.passes(measurement.outer_diameter, stddev_thresh) &&
            diff_stats.passes(measurement.diff, stddev_thresh) &&
            inner_radius_stats.passes(measurement.inner_radius, stddev_thresh) &&
            outer_radius_stats.passes(measurement.outer_radius, stddev_thresh))
        {
            cv::circle(draw_image, measurement.outer0, 3, cv::Scalar(255, 0, 0));
            cv::circle(draw_image, measurement.outer1, 3, cv::Scalar(255, 0, 0));
            cv::circle(draw_image, measurement.inner0, 3, cv::Scalar(0, 0, 255));
            cv::circle(draw_image, measurement.inner1, 3, cv::Scalar(0, 0, 255));
            cv::line(draw_image, measurement.outer0, centroid, cv::Scalar(255, 0, 0));
            cv::line(draw_image, measurement.outer1, centroid, cv::Scalar(255, 0, 0));

            cv::line(draw_image, measurement.inner0, centroid, cv::Scalar(0, 0, 255));
            cv::line(draw_image, measurement.inner1, centroid, cv::Scalar(0, 0, 255));

            auto line = Eigen::ParametrizedLine<float, 2>::Through({centroid.x, centroid.y},
                                                                   {measurement.outer0.x, measurement.outer0.y});
            auto pt = line.pointAt(measurement.outer_radius * 1.2);
            cv::putText(draw_image,
                        std::to_string(count),
                        {static_cast<int>(pt(0)), static_cast<int>(pt(1))},
                        cv::FONT_HERSHEY_COMPLEX,
                        0.8,
                        cv::Scalar(0, 255, 0),
                        2);
            ++count;
            ofs << count << ", " << measurement.inner_radius / pixel_per_um << ", "
                << measurement.outer_radius / pixel_per_um << ", " << measurement.inner_diameter / pixel_per_um << ", "
                << measurement.outer_diameter / pixel_per_um << ", " << measurement.diff / pixel_per_um << '\n';
            i += measurements.size() / 10;
        }
        else
        {
            i += 1;
        }
    }
    std::vector<std::vector<cv::Point>> contour{cell->inner_membrane, cell->outer_membrane};
    cv::drawContours(draw_image, contour, 0, cv::Scalar(0, 0, 255));
    cv::drawContours(draw_image, contour, 1, cv::Scalar(255, 0, 0));
    std::string img_path = path.string() + "_overlay.png";
    cv::imwrite(img_path, draw_image);
    overlay_param.updateData(draw_image, mo::tag::_param = image_param);
    return true;
}
}
}

using namespace aq::bio;

MO_REGISTER_CLASS(MeasureCell);
