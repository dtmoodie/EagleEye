#pragma once
#include <opencv2/core/cuda.hpp>

#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/device_ptr.h>


template<typename T> struct step_functor : public thrust::unary_function<int, int>
{
    int columns;
    int step;
    int channels;
    __host__ __device__ step_functor(int columns_, int step_, int channels_ = 1) : columns(columns_), step(step_), channels(channels_) {    };
    __host__ step_functor(cv::cuda::GpuMat& mat)
    {
        CV_Assert(mat.depth() == cv::DataType<T>::depth);
        columns = mat.cols;
        step = mat.step / sizeof(T);
        channels = mat.channels();
    }
    __host__ __device__
        int operator()(int x) const
    {
        int row = x / columns;
        int idx = (row * step) + (x % columns)*channels;
        return idx;
    }
};
template<typename T, int CN, typename PtrType> class ThrustView;
template<typename T, int CN> ThrustView<T, CN, T*> CreateView(cv::Mat& mat, int channel = 0);
template<typename T, int CN> ThrustView<const T, CN, T*> CreateView(const cv::Mat& mat, int channel = 0);

template<typename T, int CN> ThrustView<T, CN, thrust::device_ptr<T> > CreateView(cv::cuda::GpuMat& mat, int channel = 0);
template<typename T, int CN> ThrustView<const T, CN, thrust::device_ptr<const T> > CreateView(const cv::cuda::GpuMat& mat, int channel = 0);




template<typename T, int CN, typename PtrType = thrust::device_ptr<T> >
class ThrustView
{
    // Iterates over every element inside of a matrix
    typedef cv::Vec<T, CN> DataType;
    typedef thrust::permutation_iterator<PtrType, thrust::transform_iterator<step_functor<T>, thrust::counting_iterator<int> > > MatrixItr_t;
    typedef thrust::permutation_iterator<PtrType, thrust::counting_iterator<int> > RowItr_t;
    // Outer iterator iterates over each row of the matrix, inner iterator iterats over elemetns of the row
    typedef thrust::permutation_iterator<RowItr_t, thrust::transform_iterator<step_functor<T>, thrust::counting_iterator<int> > > RowsItr_t;
    typedef MatrixItr_t ColItr_t;
};

template<typename T, typename PtrType>
class ThrustView<T, 1, PtrType>
{
public:
    // Iterates over every element inside of a matrix
    typedef thrust::permutation_iterator<PtrType, thrust::transform_iterator<step_functor<T>, thrust::counting_iterator<int> > > MatrixItr_t;
    typedef thrust::permutation_iterator<PtrType, thrust::counting_iterator<int> > RowItr_t;
    typedef MatrixItr_t ColItr_t;
    // Outer iterator iterates over each row of the matrix, inner iterator iterats over elemetns of the row
    typedef thrust::permutation_iterator<thrust::zip_iterator<thrust::tuple<RowItr_t, RowItr_t> >, thrust::transform_iterator<step_functor<T>, thrust::counting_iterator<int> > > RowsItr_t;
    typedef thrust::permutation_iterator<thrust::zip_iterator<thrust::tuple<ColItr_t, ColItr_t> >, thrust::transform_iterator<step_functor<T>, thrust::counting_iterator<int> > > ColsItr_t;

    MatrixItr_t begin()
    {
        return thrust::make_permutation_iterator(PtrType(_data),
                    thrust::make_transform_iterator(thrust::make_counting_iterator(0), 
                    step_functor<T>(_cols, _row_step, _element_step)));
    }
    MatrixItr_t end()
    {
        return thrust::make_permutation_iterator(PtrType(_data),
                    thrust::make_transform_iterator(thrust::make_counting_iterator(_rows*_cols), 
                    step_functor<T>(_cols, _row_step, _element_step)));
    }

    MatrixItr_t diagBegin()
    {
        return thrust::make_permutation_iterator(PtrType(_data),
            thrust::make_transform_iterator(thrust::make_counting_iterator(0), 
                step_functor<T>(1, 1 + _row_step, _element_step)));
    }
    MatrixItr_t diagEndItr()
    {
        return thrust::make_permutation_iterator(PtrType(_data),
            thrust::make_transform_iterator(thrust::make_counting_iterator(std::min(_rows, _cols)), step_functor<T>(1, 1 + _row_step, _element_step)));
    }

    ColItr_t colBegin(const int col)
    {
        return thrust::make_permutation_iterator(PtrType(_data),
            thrust::make_transform_iterator(thrust::make_counting_iterator(0),
                step_functor<T>(1, _row_step, _element_step)));
    }
    ColItr_t colEnd(const int col)
    {
        return thrust::make_permutation_iterator(PtrType(_data),
            thrust::make_transform_iterator(thrust::make_counting_iterator(_cols),
                step_functor<T>(1, _row_step, _element_step)));
    }

    RowItr_t rowBegin(const int row)
    {
        return thrust::make_permutation_iterator(PtrType(ptr(row)),
                    thrust::make_counting_iterator(0));
    }
    RowItr_t rowEnd(const int row)
    {
        return thrust::make_permutation_iterator(PtrType(ptr(row)),
                thrust::make_counting_iterator(_cols));
    }

    std::pair<RowsItr_t, RowsItr_t> rowRange(int start_row = 0, int end_row = -1)
    {
        if(end_row == -1)
            end_row = _rows;
        auto begin_itr = rowBegin(start_row);
        auto zip = thrust::make_zip_iterator(thrust::make_tuple(begin_itr, begin_itr));

        return std::make_pair(thrust::make_permutation_iterator(zip,
                    thrust::make_transform_iterator(thrust::make_counting_iterator(0),
                        step_functor<T>(1, _row_step, _element_step))),
                    thrust::make_permutation_iterator(zip,
                        thrust::make_transform_iterator(thrust::make_counting_iterator(end_row - start_row),
                        step_functor<T>(1, _row_step, _element_step))));
    }
    // WIP
    std::pair<ColsItr_t, ColsItr_t> colRange(int start_col = 0, int end_col = -1)
    {
        if(end_col == -1)
            end_col = _cols;
        auto begin_itr = colBegin(start_col);
        auto end_itr = colEnd(start_col);
        auto zip = thrust::make_zip_iterator(thrust::make_tuple(begin_itr, end_itr));

        return std::make_pair(thrust::make_permutation_iterator(zip,
            thrust::make_transform_iterator(thrust::make_counting_iterator(0),
                step_functor<T>(1, 1, _element_step))),
            thrust::make_permutation_iterator(zip,
                thrust::make_transform_iterator(thrust::make_counting_iterator(end_col - start_col),
                    step_functor<T>(1, 1, _element_step))));
    }

    
    T* ptr(int row = 0)
    {
        return _data + _row_step * row;
    }
    T* _data;
    int _rows;
    int _cols;
    // Num elements per row, accounts for padding
    int _row_step;
    int _element_step;
};

template<typename T, int CN>
ThrustView<T, CN, T*> CreateView(cv::Mat& mat, int channel)
{
    ThrustView<T, CN, T*> view;
    CV_Assert(mat.depth() == cv::DataType<T>::depth);
    view._rows = mat.rows;
    view._cols = mat.cols;
    // Mat's step is stored in bytes
    view._row_step = mat.step / sizeof(T);
    if (channel == -1)
    {
        // Virtually reshape the matrix such that each channel of the image is seen as a separate adjacent pixel
        // IE [(RGB),(RGB)] -> [R,G,B,R,G,B]
        view._element_step = 1;
        view._data = (T*)mat.ptr();
    }
    else
    {
        view._element_step = mat.channels();
        view._data = (T*)(mat.ptr() + channel);
    }
    return view;
}

template<typename T, int CN>
ThrustView<const T, CN, T*> CreateView(const cv::Mat& mat, int channel)
{
    ThrustView<T, CN, T*> view;
    CV_Assert(mat.depth() == cv::DataType<T>::depth);
    view._rows = mat.rows;
    view._cols = mat.cols;
    // Mat's step is stored in bytes
    view._row_step = mat.step / sizeof(T);
    if (channel == -1)
    {
        // Virtually reshape the matrix such that each channel of the image is seen as a separate adjacent pixel
        // IE [(RGB),(RGB)] -> [R,G,B,R,G,B]
        view._element_step = 1;
        view._data = (T*)mat.ptr();
    }
    else
    {
        view._element_step = mat.channels();
        view._data = (T*)(mat.ptr() + channel);
    }
    return view;
}


template<typename T, int CN>
ThrustView<T, CN, thrust::device_ptr<T> > CreateView(cv::cuda::GpuMat& mat, int channel)
{
    ThrustView<T, CN, thrust::device_ptr<T> > view;
    CV_Assert(mat.depth() == cv::DataType<T>::depth);
    view._rows = mat.rows;
    view._cols = mat.cols;
    // Mat's step is stored in bytes
    view._row_step = mat.step / sizeof(T);
    if (channel == -1)
    {
        // Virtually reshape the matrix such that each channel of the image is seen as a separate adjacent pixel
        // IE [(RGB),(RGB)] -> [R,G,B,R,G,B]
        view._element_step = 1;
        view._data = (T*)mat.ptr();
    }
    else
    {
        view._element_step = mat.channels();
        view._data = mat.ptr<T>() + channel;
    }
    return view;
}

template<typename T, int CN>
ThrustView<const T, CN, thrust::device_ptr<const T> > CreateView(const cv::cuda::GpuMat& mat, int channel)
{
    ThrustView<const T, CN, thrust::device_ptr<const T> > view;
    CV_Assert(mat.depth() == cv::DataType<T>::depth);
    view._rows = mat.rows;
    view._cols = mat.cols;
    // Mat's step is stored in bytes
    view._row_step = mat.step / sizeof(T);
    if (channel == -1)
    {
        // Virtually reshape the matrix such that each channel of the image is seen as a separate adjacent pixel
        // IE [(RGB),(RGB)] -> [R,G,B,R,G,B]
        view._element_step = 1;
        view._data = mat.ptr<T>();
    }
    else
    {
        view._element_step = mat.channels();
        view._data = mat.ptr<T>() + channel;
    }
    return view;
}

/*
@Brief GpuMatBeginItr returns a thrust compatible iterator to the beginning of a GPU mat's memory.
@Param mat is the input matrix
@Param channel is the channel of the matrix that the iterator is accessing.  If set to -1, the iterator will access every element in sequential order
*/
template<typename T>
thrust::permutation_iterator<thrust::device_ptr<T>, thrust::transform_iterator<step_functor<T>, thrust::counting_iterator<int> > >  GpuMatBeginItr(cv::cuda::GpuMat& mat, int channel = 0)
{
    if (channel == -1)
    {
        mat = mat.reshape(1);
        channel = 0;
    }
    CV_Assert(mat.depth() == cv::DataType<T>::depth);
    CV_Assert(channel < mat.channels());
    return thrust::make_permutation_iterator(thrust::device_pointer_cast(mat.ptr<T>(0) + channel),
        thrust::make_transform_iterator(thrust::make_counting_iterator(0), step_functor<T>(mat.cols, mat.step / sizeof(T), mat.channels())));
}
/*
@Brief GpuMatEndItr returns a thrust compatible iterator to the end of a GPU mat's memory.
@Param mat is the input matrix
@Param channel is the channel of the matrix that the iterator is accessing.  If set to -1, the iterator will access every element in sequential order
*/
template<typename T>
thrust::permutation_iterator<thrust::device_ptr<T>, thrust::transform_iterator<step_functor<T>, thrust::counting_iterator<int> > >  GpuMatEndItr(cv::cuda::GpuMat& mat, int channel = 0)
{
    if (channel == -1)
    {
        mat = mat.reshape(1);
        channel = 0;
    }
        
    CV_Assert(mat.depth() == cv::DataType<T>::depth);
    CV_Assert(channel < mat.channels());
    return thrust::make_permutation_iterator(thrust::device_pointer_cast(mat.ptr<T>(0) + channel),
        thrust::make_transform_iterator(thrust::make_counting_iterator(mat.rows*mat.cols), step_functor<T>(mat.cols, mat.step / sizeof(T), mat.channels())));
}

template<typename T>
thrust::permutation_iterator<thrust::device_ptr<T>, thrust::transform_iterator<step_functor<T>, thrust::counting_iterator<int> > >  GpuMatDiagBeginItr(cv::cuda::GpuMat& mat, int channel = 0)
{
    if(channel == -1)
    {
        mat = mat.reshape(1);
        channel = 0;
    }
    CV_Assert(mat.depth() == cv::DataType<T>::depth);
    CV_Assert(channel < mat.channels());
    return thrust::make_permutation_iterator(thrust::device_pointer_cast(mat.ptr<T>(0) + channel),
        thrust::make_transform_iterator(thrust::make_counting_iterator(0), step_functor<T>(1, 1 + mat.step / sizeof(T), mat.channels())));
}

template<typename T>
thrust::permutation_iterator<thrust::device_ptr<T>, thrust::transform_iterator<step_functor<T>, thrust::counting_iterator<int> > >  GpuMatDiagEndItr(cv::cuda::GpuMat& mat, int channel = 0)
{
    if (channel == -1)
    {
        mat = mat.reshape(1);
        channel = 0;
    }
    CV_Assert(mat.depth() == cv::DataType<T>::depth);
    CV_Assert(channel < mat.channels());
    return thrust::make_permutation_iterator(thrust::device_pointer_cast(mat.ptr<T>(0) + channel),
        thrust::make_transform_iterator(thrust::make_counting_iterator(std::min(mat.cols, mat.rows)), step_functor<T>(1, 1 + mat.step / sizeof(T), mat.channels())));
}
