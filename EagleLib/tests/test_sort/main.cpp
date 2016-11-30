#define BOOST_TEST_MAIN

#include <opencv2/core.hpp>

#ifdef _MSC_VER
#include <boost/test/unit_test.hpp>
#else
#define BOOST_TEST_MODULE __FILE__
#include <boost/test/included/unit_test.hpp>
#endif


#include <EagleLib/utilities/GPUSorting.hpp>

BOOST_AUTO_TEST_CASE(gpu_sort_accuracy)
{
    cv::Mat h_mat(100, 100, CV_32F);
    cv::randn(h_mat, cv::Scalar(0), cv::Scalar(0.1));
    cv::cuda::GpuMat d_mat(h_mat);

    // Sort ascending all elements as if it was just one long array
    {
        cv::cuda::GpuMat d_sorted;
        cv::cuda::sort(d_mat, d_sorted, 0);
        cv::Mat h_sorted(d_sorted);
        float* data = h_sorted.ptr<float>();
        int num_elems = h_sorted.size().area();
        for(int i = 1; i < num_elems; ++i)
        {
            BOOST_REQUIRE_GE(data[i], data[i - 1]);
        }
    }
    // Sort descending
    {
        cv::cuda::GpuMat d_sorted;
        cv::cuda::sort(d_mat, d_sorted, cv::SORT_DESCENDING);
        cv::Mat h_sorted(d_sorted);
        float* data = h_sorted.ptr<float>();
        int num_elems = h_sorted.size().area();
        for (int i = 1; i < num_elems; ++i)
        {
            BOOST_REQUIRE_LE(data[i], data[i - 1]);
        }
    }
    // Sort rows
    {
        cv::cuda::GpuMat d_sorted;
        cv::cuda::sort(d_mat, d_sorted, cv::SORT_EVERY_ROW);
        cv::Mat h_sorted(d_sorted);
        int cols = h_sorted.cols;
        for(int row = 0; row < h_sorted.rows; ++row)
        {
            float* data = h_sorted.ptr<float>(row);
            for(int col = 1; col < cols; ++col)
            {
                BOOST_REQUIRE_GE(data[col], data[col - 1]);
            }
        }
    }
    // Sort descending
    {
        cv::cuda::GpuMat d_sorted;
        cv::cuda::sort(d_mat, d_sorted, cv::SORT_EVERY_ROW | cv::SORT_DESCENDING);
        cv::Mat h_sorted(d_sorted);
        int cols = h_sorted.cols;
        for (int row = 0; row < h_sorted.rows; ++row)
        {
            float* data = h_sorted.ptr<float>(row);
            for (int col = 1; col < cols; ++col)
            {
                BOOST_REQUIRE_LE(data[col], data[col - 1]);
            }
        }
    }
}


BOOST_AUTO_TEST_CASE(gpu_sort_performance)
{
    cv::Mat h_mat(100, 100, CV_32F);
    cv::randn(h_mat, cv::Scalar(0), cv::Scalar(0.1));
    cv::cuda::GpuMat d_mat(h_mat);

}

