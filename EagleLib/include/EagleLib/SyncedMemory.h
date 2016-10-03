#pragma once
#include "EagleLib/Detail/Export.hpp"
#include "EagleLib/rcc/external_includes/cv_core.hpp"
#include <opencv2/core/cuda.hpp>


namespace EagleLib
{
    template<typename T> struct FN : public T
    {
        template<class...U> FN(int frame_number_ = 0) : frame_number(frame_number_) {}
        template<class...U> FN(int frame_number_, const U&... args) : frame_number(frame_number_), T(args...) {}
        FN& operator=(const T& other)
        {
            T::operator=(other);
            return *this;
        }
        template<class A> void serialize(A& ar)
        {
            ar(frame_number);
            ar(*static_cast<T*>(this));
        }

        int frame_number;
    };
    template<typename T> struct TS: public T
    {
        template<class...U> TS(U...args):T(args...)
        {
            timestamp = 0.0;
            frame_number = 0;
        }
        template<class...U> TS(double ts, int frame_number, U...args) : T(args...)
        {
            timestamp = ts;
            this->frame_number = frame_number;
        }
        template<class A> void serialize(A& ar)
        {
            ar(frame_number);
            ar(timestamp);
            ar(*static_cast<T*>(this));
        }

        double timestamp;
        int frame_number;
    };


    class EAGLE_EXPORTS SyncedMemory
    {
        enum SYNC_STATE
        {
            SYNCED = 0,
            HOST_UPDATED,
            DEVICE_UPDATED
        };
        std::vector<cv::Mat> h_data;
        std::vector<cv::cuda::GpuMat> d_data;
        std::vector<SYNC_STATE> sync_flags;
    public:
        SyncedMemory();
        SyncedMemory(const cv::Mat& h_mat, const cv::cuda::GpuMat& d_mat);
        SyncedMemory(const cv::Mat& h_mat);
        SyncedMemory(const cv::cuda::GpuMat& d_mat);
        SyncedMemory(const std::vector<cv::cuda::GpuMat> & d_mats);
        SyncedMemory(const std::vector<cv::Mat>& h_mats);
        SyncedMemory(const std::vector<cv::Mat>& h_mat, const std::vector<cv::cuda::GpuMat>& d_mat);
        SyncedMemory(cv::MatAllocator* cpu_allocator, cv::cuda::GpuMat::Allocator* gpu_allocator);
        SyncedMemory clone(cv::cuda::Stream& stream);

        const cv::Mat&                         GetMat(cv::cuda::Stream& stream, int = 0);
        cv::Mat&                             GetMatMutable(cv::cuda::Stream& stream, int = 0);

        const cv::cuda::GpuMat&                 GetGpuMat(cv::cuda::Stream& stream, int = 0);
        cv::cuda::GpuMat&                     GetGpuMatMutable(cv::cuda::Stream& stream, int = 0);

        const std::vector<cv::Mat>&             GetMatVec(cv::cuda::Stream& stream);
        std::vector<cv::Mat>&                 GetMatVecMutable(cv::cuda::Stream& stream);

        const std::vector<cv::cuda::GpuMat>& GetGpuMatVec(cv::cuda::Stream& stream);
        std::vector<cv::cuda::GpuMat>&         GetGpuMatVecMutable(cv::cuda::Stream& stream);

        void Synchronize(cv::cuda::Stream& stream = cv::cuda::Stream::Null());
        void ResizeNumMats(int new_size = 1);
        void ReleaseGpu(cv::cuda::Stream& stream = cv::cuda::Stream::Null());

        int GetNumMats() const;
        bool empty() const;
        cv::Size GetSize() const;
        int GetChannels() const;
        std::vector<int> GetShape() const;
        int GetDim(int dim) const;
        int GetDepth() const;
        template<typename A> void load(A& ar)
        {
            
        }
        template<typename A> void save(A& ar)
        {
            
        }
    };
}