#include "snakes.hpp"
#include <Eigen/Geometry>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#define _CV_SNAKE_BIG 2.e+38f
#define _CV_SNAKE_IMAGE 1
#define _CV_SNAKE_GRAD 2

#define CV_VALUE 0
#define CV_ARRAY 1

static void icvSnake8uC1R(unsigned char* src,
                          int srcStep,
                          cv::Size roi,
                          cv::Point* pt,
                          int n,
                          float* alpha,
                          float* beta,
                          float* gamma,
                          int coeffUsage,
                          cv::Size win,
                          cv::TermCriteria criteria,
                          int scheme,
                          const float* position_weight)
{
    int i, j, k;
    int neighbors = win.height * win.width;

    int centerx = win.width >> 1;
    int centery = win.height >> 1;

    float invn;
    int iteration = 0;
    int converged = 0;

    float _alpha, _beta, _gamma;

    /*#ifdef GRAD_SNAKE */
    std::vector<float> gradient;
    std::vector<uchar> map;
    int map_width = ((roi.width - 1) >> 3) + 1;
    int map_height = ((roi.height - 1) >> 3) + 1;
#define WTILE_SIZE 8
#define TILE_SIZE (WTILE_SIZE + 2)
    short dx[TILE_SIZE * TILE_SIZE], dy[TILE_SIZE * TILE_SIZE];
    cv::Mat _dx(TILE_SIZE, TILE_SIZE, CV_16SC1, dx);
    cv::Mat _dy(TILE_SIZE, TILE_SIZE, CV_16SC1, dy);
    cv::Mat _src(roi.height, roi.width, CV_8UC1, src);

    // cv::Ptr<cv::FilterEngine> pX, pY;

    /* inner buffer of convolution process */
    // char ConvBuffer[400];

    /*#endif */

    /* check bad arguments */
    MO_ASSERT(src != nullptr);
    MO_ASSERT(roi.height > 0 && roi.width > 0);
    MO_ASSERT(srcStep >= roi.width);
    MO_ASSERT(pt != nullptr);
    MO_ASSERT(n >= 3);
    MO_ASSERT(alpha != nullptr);
    MO_ASSERT(beta != nullptr);
    MO_ASSERT(gamma != nullptr);

    MO_ASSERT(coeffUsage == CV_VALUE || coeffUsage == CV_ARRAY);
    MO_ASSERT(win.height > 0);
    MO_ASSERT(win.height & 1);
    // if ((win.height <= 0) || (!(win.height & 1)))
    //    return CV_StsBadSize;
    MO_ASSERT(win.width > 0);
    MO_ASSERT(win.width & 1);
    // if ((win.width <= 0) || (!(win.width & 1)))
    // return CV_StsBadSize;

    invn = 1 / ((float)n);

    if (scheme == _CV_SNAKE_GRAD)
    {

        // pX = cv::createDerivFilter( CV_8U, CV_16S, 1, 0, 3, cv::BORDER_REPLICATE );
        // pY = cv::createDerivFilter( CV_8U, CV_16S, 0, 1, 3, cv::BORDER_REPLICATE );
        // gradient = (float*)cvAlloc(roi.height * roi.width * sizeof(float));
        gradient.resize(roi.height * roi.width);

        // map = (uchar*)cvAlloc(map_width * map_height);
        map.resize(map_width * map_height, 0);
        /* clear map - no gradient computed */
        // memset((void*)map, 0, map_width * map_height);
    }
    std::vector<float> Econt(neighbors);
    std::vector<float> Ecurv(neighbors);
    std::vector<float> Eimg(neighbors);
    std::vector<float> E(neighbors);

    while (!converged)
    {
        float ave_d = 0;
        int moved = 0;

        converged = 0;
        iteration++;
        /* compute average distance */
        for (i = 1; i < n; i++)
        {
            int diffx = pt[i - 1].x - pt[i].x;
            int diffy = pt[i - 1].y - pt[i].y;

            ave_d += std::sqrt((float)(diffx * diffx + diffy * diffy));
        }
        ave_d += std::sqrt((float)((pt[0].x - pt[n - 1].x) * (pt[0].x - pt[n - 1].x) +
                                   (pt[0].y - pt[n - 1].y) * (pt[0].y - pt[n - 1].y)));

        ave_d *= invn;
        /* average distance computed */
        for (i = 0; i < n; i++)
        {
            /* Calculate Econt */
            float maxEcont = 0;
            float maxEcurv = 0;
            float maxEimg = 0;
            float minEcont = _CV_SNAKE_BIG;
            float minEcurv = _CV_SNAKE_BIG;
            float minEimg = _CV_SNAKE_BIG;
            float Emin = _CV_SNAKE_BIG;

            int offsetx = 0;
            int offsety = 0;
            float tmp;

            /* compute bounds */
            int left = MIN(pt[i].x, win.width >> 1);
            int right = MIN(roi.width - 1 - pt[i].x, win.width >> 1);
            int upper = MIN(pt[i].y, win.height >> 1);
            int bottom = MIN(roi.height - 1 - pt[i].y, win.height >> 1);

            maxEcont = 0;
            minEcont = _CV_SNAKE_BIG;
            for (j = -upper; j <= bottom; j++)
            {
                for (k = -left; k <= right; k++)
                {
                    int diffx, diffy;
                    float energy;

                    if (i == 0)
                    {
                        diffx = pt[n - 1].x - (pt[i].x + k);
                        diffy = pt[n - 1].y - (pt[i].y + j);
                    }
                    else
                    {
                        diffx = pt[i - 1].x - (pt[i].x + k);
                        diffy = pt[i - 1].y - (pt[i].y + j);
                    }
                    Econt[(j + centery) * win.width + k + centerx] = energy =
                        (float)fabs(ave_d - std::sqrt((float)(diffx * diffx + diffy * diffy)));

                    maxEcont = MAX(maxEcont, energy);
                    minEcont = MIN(minEcont, energy);
                }
            }
            tmp = maxEcont - minEcont;
            tmp = (tmp == 0) ? 0 : (1 / tmp);
            for (k = 0; k < neighbors; k++)
            {
                Econt[k] = (Econt[k] - minEcont) * tmp;
            }

            /*  Calculate Ecurv */
            maxEcurv = 0;
            minEcurv = _CV_SNAKE_BIG;
            for (j = -upper; j <= bottom; j++)
            {
                for (k = -left; k <= right; k++)
                {
                    int tx, ty;
                    float energy;

                    if (i == 0)
                    {
                        tx = pt[n - 1].x - 2 * (pt[i].x + k) + pt[i + 1].x;
                        ty = pt[n - 1].y - 2 * (pt[i].y + j) + pt[i + 1].y;
                    }
                    else if (i == n - 1)
                    {
                        tx = pt[i - 1].x - 2 * (pt[i].x + k) + pt[0].x;
                        ty = pt[i - 1].y - 2 * (pt[i].y + j) + pt[0].y;
                    }
                    else
                    {
                        tx = pt[i - 1].x - 2 * (pt[i].x + k) + pt[i + 1].x;
                        ty = pt[i - 1].y - 2 * (pt[i].y + j) + pt[i + 1].y;
                    }
                    Ecurv[(j + centery) * win.width + k + centerx] = energy = (float)(tx * tx + ty * ty);
                    maxEcurv = MAX(maxEcurv, energy);
                    minEcurv = MIN(minEcurv, energy);
                }
            }
            tmp = maxEcurv - minEcurv;
            tmp = (tmp == 0) ? 0 : (1 / tmp);
            for (k = 0; k < neighbors; k++)
            {
                Ecurv[k] = (Ecurv[k] - minEcurv) * tmp;
            }

            /* Calculate Eimg */
            for (j = -upper; j <= bottom; j++)
            {
                for (k = -left; k <= right; k++)
                {
                    float energy;

                    if (scheme == _CV_SNAKE_GRAD)
                    {
                        /* look at map and check status */
                        int x = (pt[i].x + k) / WTILE_SIZE;
                        int y = (pt[i].y + j) / WTILE_SIZE;

                        if (map[y * map_width + x] == 0)
                        {
                            int l, m;

                            /* evaluate block location */
                            int upshift = y ? 1 : 0;
                            int leftshift = x ? 1 : 0;
                            int bottomshift = MIN(1, roi.height - (y + 1) * WTILE_SIZE);
                            int rightshift = MIN(1, roi.width - (x + 1) * WTILE_SIZE);
                            cv::Rect g_roi = {x * WTILE_SIZE - leftshift,
                                              y * WTILE_SIZE - upshift,
                                              leftshift + WTILE_SIZE + rightshift,
                                              upshift + WTILE_SIZE + bottomshift};

                            cv::Mat _src_ = _src(g_roi);
                            cv::Mat _dx_ = _dx;
                            cv::Mat _dy_ = _dy;
                            cv::Sobel(_src_, _dx_, CV_16S, 1, 0, 3, 1.0, 0, cv::BORDER_REPLICATE);
                            cv::Sobel(_src_, _dy_, CV_16S, 0, 1, 3, 1.0, 0, cv::BORDER_REPLICATE);

                            for (l = 0; l < WTILE_SIZE + bottomshift; l++)
                            {
                                for (m = 0; m < WTILE_SIZE + rightshift; m++)
                                {
                                    gradient[(y * WTILE_SIZE + l) * roi.width + x * WTILE_SIZE + m] =
                                        (float)(dx[(l + upshift) * TILE_SIZE + m + leftshift] *
                                                    dx[(l + upshift) * TILE_SIZE + m + leftshift] +
                                                dy[(l + upshift) * TILE_SIZE + m + leftshift] *
                                                    dy[(l + upshift) * TILE_SIZE + m + leftshift]);
                                }
                            }
                            map[y * map_width + x] = 1;
                        }
                        energy = gradient[(pt[i].y + j) * roi.width + pt[i].x + k];
                        if (position_weight && j == 0 && k == 0)
                            energy *= position_weight[n];
                        Eimg[(j + centery) * win.width + k + centerx] = energy;
                    }
                    else
                    {
                        Eimg[(j + centery) * win.width + k + centerx] = energy =
                            src[(pt[i].y + j) * srcStep + pt[i].x + k];
                    }

                    maxEimg = MAX(maxEimg, energy);
                    minEimg = MIN(minEimg, energy);
                }
            }

            tmp = (maxEimg - minEimg);
            tmp = (tmp == 0) ? 0 : (1 / tmp);

            for (k = 0; k < neighbors; k++)
            {
                Eimg[k] = (minEimg - Eimg[k]) * tmp;
            }

            /* locate coefficients */
            if (coeffUsage == CV_VALUE)
            {
                _alpha = *alpha;
                _beta = *beta;
                _gamma = *gamma;
            }
            else
            {
                _alpha = alpha[i];
                _beta = beta[i];
                _gamma = gamma[i];
            }

            /* Find Minimize point in the neighbors */
            for (k = 0; k < neighbors; k++)
            {
                E[k] = _alpha * Econt[k] + _beta * Ecurv[k] + _gamma * Eimg[k];
            }
            Emin = _CV_SNAKE_BIG;
            for (j = -upper; j <= bottom; j++)
            {
                for (k = -left; k <= right; k++)
                {

                    if (E[(j + centery) * win.width + k + centerx] < Emin)
                    {
                        Emin = E[(j + centery) * win.width + k + centerx];
                        offsetx = k;
                        offsety = j;
                    }
                }
            }

            if (offsetx || offsety)
            {
                pt[i].x += offsetx;
                pt[i].y += offsety;
                moved++;
            }
        }
        converged = (moved == 0);
        if ((criteria.type & cv::TermCriteria::MAX_ITER) && (iteration >= criteria.maxCount))
        {
            converged = 1;
        }

        if ((criteria.type & cv::TermCriteria::EPS) && (moved <= criteria.epsilon))
        {
            converged = 1;
        }
    }
    return;
}

namespace aqcore
{

    void sampleCircle(mt::Tensor<cv::Point, 1> pts, aq::Circle<float> circle, float pad)
    {
        const uint32_t samples = pts.getShape()[0];
        for (size_t i = 0; i < samples; ++i)
        {
            double theta = (2.0 / static_cast<double>(samples) * M_PI) * i;

            pts[i].x = static_cast<int>(circle.origin(0) + pad * circle.radius * cos(theta));
            pts[i].y = static_cast<int>(circle.origin(1) + pad * circle.radius * sin(theta));
        }
    }
    void sampleCircle(std::vector<cv::Point>& pts, aq::Circle<float> circle, float pad, int samples)
    {
        pts.resize(samples);
        sampleCircle(mt::tensorWrap(pts), circle, pad);
    }

    bool snakePoints(const cv::Mat& img,
                     std::vector<cv::Point>& points,
                     int kernel_size,
                     cv::TermCriteria term_crit,
                     int mode,
                     float* alpha,
                     float* beta,
                     float* gamma,
                     int coefficients,
                     const float* position_weight)
    {
        uchar* src = const_cast<uchar*>(img.ptr<uchar>());
        int step = img.step;
        cv::Size size(img.size());
        icvSnake8uC1R(src,
                      step,
                      size,
                      points.data(),
                      points.size(),
                      alpha,
                      beta,
                      gamma,
                      coefficients == 1 ? CV_VALUE : CV_ARRAY,
                      cv::Size(kernel_size, kernel_size),
                      term_crit,
                      mode,
                      position_weight);
        return true;
    }

    bool SnakeCircle::snakePoints(const cv::Mat& img, std::vector<cv::Point>& points)
    {
        cv::TermCriteria criteria(cv::TermCriteria::MAX_ITER, iterations, 0.1);
        int mode = this->mode.getValue();

        return aqcore::snakePoints(img, points, window_size, criteria, mode, &alpha, &beta, &gamma);
    }

    bool SnakeCircle::processImpl()
    {
        mo::IAsyncStreamPtr_t stream = this->getStream();
        const cv::Mat mat = input->getMat(stream.get());

        aq::TEntityComponentSystem<aq::Contour> ecs = *circles;
        mt::Tensor<const aq::Circlef, 1> circle_view = ecs.getComponent<aq::CircleComponent>();
        const uint32_t num_elems = circles->getNumEntities();
        ecs.resize(num_elems);
        std::vector<cv::Point> pts;
        auto contours = ecs.getComponentMutable<aq::Contour>();
        for (uint32_t i = 0; i < num_elems; ++i)
        {
            const aq::Circlef& circle = circle_view[i];

            sampleCircle(pts, circle, pad, num_samples);
            if (snakePoints(mat, pts))
            {
                contours[i] = pts;
            }
            else
            {
                ecs.erase(i);
            }
        }
        output.publish(std::move(ecs), mo::tags::param = &input_param);
        return true;
    }
} // namespace aqcore

#include <Aquila/nodes/NodeInfo.hpp>
using namespace aqcore;
MO_REGISTER_CLASS(SnakeCircle)
