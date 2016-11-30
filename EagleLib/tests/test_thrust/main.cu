#include <EagleLib/Thrust_interop.hpp>
#include <opencv2/core.hpp>
#include <thrust/sequence.h>
#include <thrust/sort.h>

template<typename T> struct UnarySortDescending
{
    template<typename U1> void operator()(const U1& it1)
    {
        thrust::sort(&thrust::get<0>(it1), &thrust::get<1>(it1), thrust::greater<T>());
    }
};


template<typename T> struct UnarySortAscending
{
    template<typename U1> void operator()(const U1& it1)
    {
        thrust::sort(&thrust::get<0>(it1), &thrust::get<1>(it1), thrust::less<T>());
    }
};

int main()
{
    cv::Mat image(100,100, CV_32F);
    auto view = CreateView<float, 1>(image);
    thrust::sequence(view.begin(), view.end());
    thrust::sort(view.rowBegin(10), view.rowEnd(10), thrust::greater<float>());
    auto range = view.rowRange(5, 15);
    thrust::for_each(range.first, range.second, UnarySortDescending<float>());
    // Sort a single column
    thrust::sort(view.colBegin(0), view.colEnd(0), thrust::greater<float>());

    auto col_range = view.colRange(15, 30);
    thrust::for_each(col_range.first, col_range.second, UnarySortDescending<float>());
    return 0;
}