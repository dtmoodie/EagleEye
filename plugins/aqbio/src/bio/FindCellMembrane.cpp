#include "FindCellMembrane.hpp"
#include <Aquila/nodes/NodeInfo.hpp>
#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <random>


namespace aq
{

double medianMat(cv::Mat Input, int nVals)
{
    // COMPUTE HISTOGRAM OF SINGLE CHANNEL MATRIX
    float range[] = {0, float(nVals)};
    const float* histRange = {range};
    bool uniform = true;
    bool accumulate = false;
    cv::Mat hist;
    cv::calcHist(&Input, 1, 0, cv::Mat(), hist, 1, &nVals, &histRange, uniform, accumulate);

    // COMPUTE CUMULATIVE DISTRIBUTION FUNCTION (CDF)
    cv::Mat cdf;
    hist.copyTo(cdf);
    for (int i = 1; i <= nVals - 1; i++)
    {
        cdf.at<float>(i) += cdf.at<float>(i - 1);
    }
    cdf /= Input.total();

    // COMPUTE MEDIAN
    double medianVal;
    for (int i = 0; i <= nVals - 1; i++)
    {
        if (cdf.at<float>(i) >= 0.5)
        {
            medianVal = i;
            break;
        }
    }
    return medianVal / nVals;
}

template <class T>
void drawSeam(const cv::Mat_<T>& energy_, cv::Mat_<int>& seam)
{
    cv::Mat_<T> energy = energy_;
    for (int i = 0; i < energy.rows; ++i)
    {
        energy(i, seam(i, 0)) = 0;
    }
}

template <class T>
void findMaximalSeam(const cv::Mat_<T>& energy, cv::Mat_<int>& seam, int search_window)
{
    const int nrows = energy.rows;
    const int ncols = energy.cols;
    cv::Mat_<int> selection(nrows, ncols);
    selection.setTo(0);
    cv::Mat_<T> accumulation(nrows, ncols);
    accumulation.setTo(0);

    int search_step = search_window / 2;
    energy.row(0).copyTo(accumulation.row(0));

    for (int row = 1; row < nrows; ++row)
    {
        for (int col = 0; col < ncols; ++col)
        {
            int x = std::max(0, col - search_step);
            int max_idx = x;
            T max_val = accumulation(row - 1, x);
            ++x;
            int end = std::min(ncols, col + search_step);
            for (; x < end; ++x)
            {
                if (accumulation(row - 1, x) > max_val)
                {
                    max_val = accumulation(row - 1, x);
                    max_idx = x;
                }
            }
            selection(row, col) = max_idx;
            accumulation(row, col) = max_val + energy(row, col);
        }
    }
    // Find the highest accumulation in the last row, and then work backwards
    int max_idx;
    T* ptr = accumulation.template ptr<T>(nrows - 1);
    T* max_elem = std::max_element(ptr, ptr + ncols);
    max_idx = max_elem - ptr;
    seam.create(nrows, 1);

    for (int i = nrows - 1; i >= 0; --i)
    {
        seam(i, 0) = max_idx;
        max_idx = selection(i, max_idx);
    }
}

void findMaximalSeam(const cv::Mat& /*score*/, cv::Mat& /*seam*/, int /*search_window*/)
{
    // findMaximalSeam(cv::Mat_<short>(score), seam, search_window);
}

namespace bio
{

size_t revIndex(const ssize_t size, const ssize_t idx)
{
    if(idx < 0)
    {
        return size - idx;
    }else
    {
        return idx;
    }
}

float gaussianDist(const float sigma, const float mean, const float x)
{
    return ( 1.0F / ( sigma * sqrt(2.0F*M_PI) ) ) * exp( -0.5F * pow( (x-mean)/sigma, 2.0F ) );
}

float gaussianDist(const float sigma,  const float x)
{
    return gaussianDist(sigma, 0.0F, x);
}


float Cell::dist(const cv::Point& pt) const
{
    const float dx = pt.x - center.x();
    const float dy = pt.y - center.y();
    return sqrt(dx * dx + dy * dy);
}

void Cell::push(cv::Point& pt, const float dist)
{
    Eigen::ParametrizedLine<float, 2> ray =
        Eigen::ParametrizedLine<float, 2>::Through(center, {pt.x, pt.y});
    const auto radius = (center - Eigen::Vector2f(pt.x, pt.y)).norm();
    const auto new_pt = ray.pointAt(radius + dist);
    pt.x = new_pt.x();
    pt.x = new_pt.y();
}

void Cell::pushInner(const ssize_t idx, const float dist)
{
    push(inner_membrane[revIndex(inner_membrane.size(), idx)], dist);
}

void Cell::pushOuter(const ssize_t idx, const float dist)
{
    push(outer_membrane[revIndex(outer_membrane.size(), idx)], dist);
}

void Cell::clear()
{
    inner_membrane.clear();
    outer_membrane.clear();
    inner_point_position_confidence.clear();
    outer_point_position_confidence.clear();
    inner_updated = false;
    outer_updated = false;
    fn = -1;
}

void FindCellMembrane::mouseDrag(std::string /*window_name*/, cv::Point start, cv::Point end, int /*flags*/, cv::Mat /*img*/)
{
    // Look through inner and outer cell points to find closest to start
    bool closest_is_inner = true;
    float dist = std::numeric_limits<float>::max();
    cv::Point* pt = nullptr;
    for (auto& inner_pt : m_current_cell.inner_membrane)
    {
        float d = cv::norm(start - inner_pt);
        if (d < dist)
        {
            dist = d;
            pt = &inner_pt;
        }
    }
    
    for (auto& outer_pt : m_current_cell.outer_membrane)
    {
        float d = cv::norm(start - outer_pt);
        if (d < dist)
        {
            dist = d;
            pt = &outer_pt;
            closest_is_inner = false;
        }
    }
    if (dist > 5)
        return;
    if (closest_is_inner)
    {
        ssize_t idx = pt - &m_current_cell.inner_membrane[0];
        m_current_cell.inner_point_position_confidence[idx] = 10.0;
        m_current_cell.inner_updated = true;

        const float r1 = m_current_cell.dist(end);
        const float r2 = m_current_cell.dist(m_current_cell.inner_membrane[idx]);
        const float dist = r1 - r2;

        m_current_cell.inner_membrane[idx] = end;

        for(ssize_t i = idx - 3; i < idx + 3; ++i)
        {
            m_current_cell.pushInner(i, dist);
        }
    }
    else
    {
        ssize_t idx = pt - &m_current_cell.outer_membrane[0];
        m_current_cell.outer_point_position_confidence[idx] = 10.0;
        m_current_cell.outer_updated = true;

        /*const float r1 = m_current_cell.dist(end);
        const float r2 = m_current_cell.dist(m_current_cell.outer_membrane[idx]);
        const float dist = r1 - r2;*/

        m_current_cell.outer_membrane[idx] = end;

        for(ssize_t i = idx - 3; i < idx + 3; ++i)
        {
            //const float scale = gaussianDist(mouse_sigma, i - idx);
            //m_current_cell.pushOuter(i, dist);
        }

        *pt = end;
    }

    setModified();
    sig_update();
    user_update_param.updateData(true);
}

std::vector<cv::Point> FindCellMembrane::findOuterMembrane(const cv::Mat& img,
                                                           const std::vector<cv::Point>& inner,
                                                           const aq::Circle<float>& circle)
{
    // now look for outer cell wall
    cv::Mat_<short> dx, dy;
    cv::Sobel(img, dx, CV_16S, 1, 0, kernel_size, 1.0, 0.0, cv::BORDER_REPLICATE);
    cv::Sobel(img, dy, CV_16S, 0, 1, kernel_size, 1.0, 0.0, cv::BORDER_REPLICATE);
    boost::accumulators::accumulator_set<float,
                                         boost::accumulators::features<boost::accumulators::tag::mean,
                                                                       boost::accumulators::tag::variance,
                                                                       boost::accumulators::tag::min,
                                                                       boost::accumulators::tag::max>>
        Racc;
    std::vector<float> R;
    std::vector<Eigen::ParametrizedLine<float, 2>> rays;
    for (size_t i = 0; i < inner.size(); ++i)
    {
        // Eigen::Vector2f dir(cos(i), sin(i));
        Eigen::ParametrizedLine<float, 2> ray =
            Eigen::ParametrizedLine<float, 2>::Through(circle.origin, {inner[i].x, inner[i].y});
        std::vector<float> D;
        int end = 0;
        boost::accumulators::accumulator_set<
            float,
            boost::accumulators::features<boost::accumulators::tag::min, boost::accumulators::tag::max>>
            Dacc;

        for (float x = circle.radius * outer_pad; x > circle.radius * 0.9f; x -= 0.5f)
        {
            auto pt = ray.pointAt(x);
            if (pt(0) >= 0 && pt(0) < dx.cols && pt(1) >= 0 && pt(1) < dx.rows)
            {
                float d =
                    abs(ray.direction().dot(Eigen::Vector2f(dx.at<short>(pt[1], pt[0]), dy.at<short>(pt[1], pt[0]))));
                Dacc(d);
                D.push_back(d);
                if (end == 0 && x < circle.radius * inner_pad)
                {
                    end = D.size();
                }
            }
        }
        float min = boost::accumulators::min(Dacc);
        float range = boost::accumulators::max(Dacc) - min;
        std::transform(D.begin(), D.end(), D.begin(), [range, min](float val) { return (val - min) / range; });
        for (size_t j = 0; j < end; ++j)
        {
            if (D[j] > outer_threshold)
            {
                rays.push_back(ray);
                float r = circle.radius * outer_pad - j * 0.5f;
                R.push_back(r);
                Racc(r);
                break;
            }
        }
    }
    float mean = boost::accumulators::mean(Racc);
    float stddev = std::sqrt(boost::accumulators::variance(Racc));
    std::vector<cv::Point> pts;
    for (size_t i = 0; i < R.size(); ++i)
    {
        if (R[i] < mean + stddev && R[i] > mean - stddev)
        {
            auto pt = rays[i].pointAt(R[i]);
            pts.emplace_back(pt(0), pt(1));
        }
    }
    return pts;
}

float sigmoid(float x)
{
    return 1.0f / (1.0f + std::exp(-x));
}

std::vector<cv::Point> FindCellMembrane::findOuterMembraneDP(const cv::Mat& img,
                                                             const std::vector<cv::Point>& inner,
                                                             const aq::Circle<float>& circle)
{
    size_t num_samples = inner.size();
    size_t num_positions = (circle.radius * outer_pad - circle.radius * inner_pad) / radial_resolution;
    cv::Mat_<short> dx, dy;
    cv::Sobel(img, dx, CV_16S, 1, 0, kernel_size, 1.0, 0.0, cv::BORDER_REPLICATE);
    cv::Sobel(img, dy, CV_16S, 0, 1, kernel_size, 1.0, 0.0, cv::BORDER_REPLICATE);

    cv::Mat_<float> energy(num_samples, num_positions);
    energy.setTo(0);
    for (size_t i = 0; i < num_samples; ++i)
    {
        Eigen::ParametrizedLine<float, 2> ray =
            Eigen::ParametrizedLine<float, 2>::Through(circle.origin, {inner[i].x, inner[i].y});
        float r = (circle.origin - Eigen::Vector2f(inner[i].x, inner[i].y)).norm();
        for (int j = 0; j < num_positions; ++j)
        {
            float x = r * inner_pad + j * radial_resolution;
            auto pt = ray.pointAt(x);
            if (pt(0) >= 0 && pt(0) < dx.cols && pt(1) >= 0 && pt(1) < dx.rows)
            {
                float d =
                    abs(ray.direction().dot(Eigen::Vector2f(dx.at<short>(pt[1], pt[0]), dy.at<short>(pt[1], pt[0]))));
                energy(i, j) = d; // * sigmoid(radial_weight * (float(j) - mid) / float(num_positions));
            }
        }
    }
    reweightEnergy(energy);
    cv::Mat_<int> seam;
    findMaximalSeam<float>(energy, seam, 5);
    drawSeam(energy, seam);
    std::vector<cv::Point> pts;
    pts.reserve(seam.rows);
    for (int i = 0; i < seam.rows; ++i)
    {
        Eigen::ParametrizedLine<float, 2> ray =
            Eigen::ParametrizedLine<float, 2>::Through(circle.origin, {inner[i].x, inner[i].y});
        float r = (circle.origin - Eigen::Vector2f(inner[i].x, inner[i].y)).norm();
        auto pt = ray.pointAt(seam(i) * radial_resolution + r * inner_pad);
        pts.emplace_back(pt(0), pt(1));
    }
    return pts;
}

void FindCellMembrane::reweightEnergy(cv::Mat_<float>& energy)
{
    // cv::Mat_<float> sorted_energy = energy.clone();
    // std::sort(sorted_energy.ptr<float>(), sorted_energy.ptr<float>() + sorted_energy.total());
    // float median = sorted_energy(sorted_energy.total() / 2);
    std::vector<double> mean, stddev;
    cv::meanStdDev(energy, mean, stddev);
    float threshold = static_cast<float>(mean[0]); // + stddev[0]);
    // for each line weight the observations based on passing a threshold, starting from the outside and moving inwards.
    // Weight outer observations more
    for (int row = 0; row < energy.rows; ++row)
    {
        float pixel_peak_count = 0.0f;
        for (int col = energy.cols - 1; col >= 0; --col)
        {
            if (energy(row, col) > threshold)
                pixel_peak_count += radial_weight;
        }
        float current_count = pixel_peak_count;
        for (int col = energy.cols - 1; col >= 0; --col)
        {
            if (energy(row, col) > threshold)
            {
                energy(row, col) *= current_count / pixel_peak_count;
                current_count -= radial_weight;
            }
            else
            {
                energy(row, col) = 0;
            }
        }
    }
}

bool FindCellMembrane::snakePoints(const cv::Mat& img, std::vector<cv::Point>& points, const std::vector<float>& point_weight)
{
    MO_ASSERT_EQ(points.size(), point_weight.size());
    return aq::snakePoints(img,
        points,
        window_size,
        cvTermCriteria(CV_TERMCRIT_ITER, iterations, 0.1),
        mode.getValue(),
        &alpha,
        &beta,
        &gamma,
        1, point_weight.data());
}

bool FindCellMembrane::processImpl()
{
    const cv::Mat& mat = input->getMat(_ctx.get());
    output.clear();
    if (!circles->empty())
    {
        auto largest = std::max_element(circles->begin(),
                                        circles->end(),
                                        [](const Detection<Circle<float>>& i1, const Detection<Circle<float>>& i2) {
                                            return i1.confidence < i2.confidence;
                                        });
        if (largest->radius > 10)
        {

            std::vector<cv::Point> inner_pts;
            std::vector<cv::Point> outer_pts;
            bool refine_inner = true;
            bool refine_outer = true;
            bool find_outer = true;
            if (user_update)
            {
                refine_inner = false;
                refine_outer = false;
                find_outer = false;
                inner_pts = m_current_cell.inner_membrane;
                outer_pts = m_current_cell.outer_membrane;
                if (!m_current_cell.inner_updated)
                    refine_inner = false;
                if (m_current_cell.outer_updated)
                    refine_outer = true;
                user_update = false;
                user_update_param.modified(false);
            }
            else
            {
                m_current_cell.clear();
                sampleCircle(inner_pts, *largest, 1.0f, num_samples);
                m_current_cell.center = largest->origin;
            }
            
            if (refine_inner)
            {
                if (m_current_cell.inner_updated == false)
                {
                    m_current_cell.inner_point_position_confidence = std::vector<float>(inner_pts.size(), 1.0f);
                }
                if (snakePoints(mat, inner_pts, m_current_cell.inner_point_position_confidence))
                {
                    output.push_back(inner_pts);
                    m_current_cell.inner_membrane = std::move(inner_pts);
                }
            }
            else
            {
                output.push_back(inner_pts);
            }

            if (find_outer)
            {
                if (method.getValue() == Naive)
                {
                    outer_pts = findOuterMembrane(mat, m_current_cell.inner_membrane, *largest);
                }
                else
                {
                    outer_pts = findOuterMembraneDP(mat, m_current_cell.inner_membrane, *largest);
                }
            }
            if (refine_outer)
            {
                if (m_current_cell.outer_updated == false)
                    m_current_cell.outer_point_position_confidence = std::vector<float>(outer_pts.size(), 1.0f);
                if (snakePoints(mat, outer_pts, m_current_cell.outer_point_position_confidence))
                {
                    output.push_back(outer_pts);
                    m_current_cell.outer_membrane = std::move(outer_pts);
                    m_current_cell.fn = input_param.getFrameNumber();
                    cell_param.updateData(m_current_cell, mo::tag::_param = input_param);
                }
            }
        }
    }
    output_param.emitUpdate(input_param);
    return true;
}
}
}

using namespace aq::bio;
MO_REGISTER_CLASS(FindCellMembrane)
