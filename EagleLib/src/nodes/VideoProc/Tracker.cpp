#include "nodes/VideoProc/Tracker.h"
#include "nodes/VideoProc/Tracking.hpp"
#include "opencv2/imgproc.hpp"
#include <opencv2/calib3d.hpp>
using namespace EagleLib;
NODE_DEFAULT_CONSTRUCTOR_IMPL(KeyFrameTracker)
NODE_DEFAULT_CONSTRUCTOR_IMPL(CMTTracker)
NODE_DEFAULT_CONSTRUCTOR_IMPL(TLDTracker)



void KeyFrameTracker::Init(bool firstInit)
{
    if(firstInit)
    {
        updateParameter("Number of key frames to track", int(5));   // 0
        updateParameter("Tracking Upper qualty", double(0.7));      // 1
        updateParameter("Tracking lower quality", double(0.4));     // 2
        updateParameter("Min key points", int(200));                // 3
        addInputParameter<DetectAndComputeFunctor>("Detector");     // 4
        addInputParameter<TrackSparseFunctor>("Tracker");
        addInputParameter<cv::cuda::GpuMat>("Mask");
        addInputParameter<int>("Index");
        addInputParameter<boost::function<void(cv::cuda::GpuMat, cv::cuda::GpuMat,
                                               cv::cuda::GpuMat, cv::cuda::GpuMat,
                                               std::string&, cv::cuda::Stream)>>("Display functor");
        homographyBuffer.resize(20);
        trackedFrames.set_capacity(5);
    }

}
void KeyFrameTracker_findHomographyCallback(int status, void* userData)
{
    TrackingResults* results = (TrackingResults*)userData;
    cv::Mat mask, refPts, trackedPts;
    cv::Mat finalMask = results->h_status.createMatHeader();
    // Pre filter points
    std::vector<int> idxMap;
    if(results->preFilter)
    {
        int goodPts = cv::countNonZero(results->h_status);
        idxMap.reserve(goodPts);
        refPts = cv::Mat(goodPts, 1, CV_32FC2);
        trackedPts = cv::Mat(goodPts, 1, CV_32FC2);
        uchar* maskPtr = finalMask.ptr<uchar>(0);
        cv::Mat refPts_ = results->h_keyFramePts.createMatHeader();
        cv::Mat trackedPts_ = results->h_trackedFramePts.createMatHeader();
        for(int i = 0; i < results->h_keyFramePts.cols; ++i, ++maskPtr)
        {
            if(*maskPtr)
            {
                trackedPts.at<cv::Vec2f>(i) = trackedPts_.at<cv::Vec2f>(i);
                refPts.at<cv::Vec2f>(i) = refPts_.at<cv::Vec2f>(i);
                idxMap.push_back(i);
            }
        }
    }else
    {
        refPts = results->h_keyFramePts.createMatHeader();
        trackedPts = results->h_trackedFramePts.createMatHeader();
    }
    cv::findHomography(refPts, trackedPts, cv::RANSAC, 3, mask, 2000, 0.995);
    // Post filter based on ransac inliers

    finalMask = cv::Scalar(0);
    for(int i = 0; i < mask.cols; ++i)
    {
        if(idxMap.size())
        {
            if(mask.at<uchar>(i))
            {
                finalMask.at<uchar>(idxMap[i]) = 255;
            }
        }else
        {
            finalMask.at<uchar>(i) = 255;
        }
    }
    std::cout << 2 << " " << clock() << std::endl;
}

cv::cuda::GpuMat KeyFrameTracker::doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream)
{
    if(parameters[0]->changed)
    {
        trackedFrames.set_capacity(getParameter<int>(0)->data);
        parameters[0]->changed = false;
    }
    DetectAndComputeFunctor* detector = getParameter<DetectAndComputeFunctor*>("Detector")->data;
    TrackSparseFunctor* tracker = getParameter<TrackSparseFunctor*>("Tracker")->data;
    cv::cuda::GpuMat* mask = getParameter<cv::cuda::GpuMat*>("Mask")->data;
    boost::function<void(cv::cuda::GpuMat, cv::cuda::GpuMat,
                         cv::cuda::GpuMat, cv::cuda::GpuMat,
                         std::string&, cv::cuda::Stream)>*
        display = getParameter<boost::function<void(cv::cuda::GpuMat, cv::cuda::GpuMat,
                                                    cv::cuda::GpuMat, cv::cuda::GpuMat,
                                                    std::string&, cv::cuda::Stream)>*>("Display functor")->data;
    int* index = getParameter<int*>("Index")->data;
    static int frameCount = 0;
    if(index)
        frameCount = *index;

    if(detector && tracker)
    {
        if(trackedFrames.empty())
        {
            TrackedFrame tf(img,frameCount);
            cv::cuda::GpuMat& keyPoints = tf.keyFrame.getKeyPoints();
            (*detector)(img,
                        mask? *mask: cv::cuda::GpuMat(),
                        keyPoints,
                        tf.keyFrame.getDescriptors(),
                        stream);
            if(keyPoints.cols > getParameter<int>(3)->data)
                trackedFrames.push_back(tf);
        }else
        {
            // Track this frame relative to all of the tracked frames
            static std::vector<cv::cuda::Stream> workStreams;
            static cv::cuda::Event startEvent;
            static std::vector<cv::cuda::Event> workFinishedEvents;
            startEvent.record(stream);
            workStreams.resize(trackedFrames.size());
            workFinishedEvents.resize(trackedFrames.size());

            int i = 0;
            for(auto itr = trackedFrames.begin(); itr != trackedFrames.end(); ++itr, ++i)
            {
                workStreams[i].waitEvent(startEvent);

                (*tracker)(itr->keyFrame.img, img,
                           itr->keyFrame.getKeyPoints(),
                           itr->trackedPoints,
                           itr->status,
                           itr->error, workStreams[i]);
                EventBuffer<TrackingResults>* h_buffer = homographyBuffer.getFront();
                h_buffer->data.KeyFrameIdx = itr->keyFrame.frameIndex;
                h_buffer->data.TrackedFrameIdx = frameCount;
                h_buffer->data.d_keyFramePts = itr->keyFrame.getKeyPoints();
                h_buffer->data.d_status = itr->status;
                h_buffer->data.d_trackedFramePts = itr->trackedPoints;
                h_buffer->data.d_keyFramePts.download(h_buffer->data.h_keyFramePts, workStreams[i]);
                h_buffer->data.d_status.download(h_buffer->data.h_status, workStreams[i]);
                h_buffer->data.d_trackedFramePts.download(h_buffer->data.h_trackedFramePts, workStreams[i]);
                h_buffer->data.preFilter = true;
                h_buffer->fillEvent.record(workStreams[i]);
            }
            i = 0;
            for(auto itr = trackedFrames.begin(); itr != trackedFrames.end(); ++itr, ++i)
            {
                std::cout << "Wait cycles: " << clock() << " ";
                EventBuffer<TrackingResults>* h_buffer = homographyBuffer.waitBack();
                std::cout << clock() << std::endl;
                KeyFrameTracker_findHomographyCallback(0, (void*)&h_buffer->data);
            }
        }
    }
    std::cout << ++frameCount << std::endl;
    return img;
}

void CMTTracker::Init(bool firstInit)
{

}

cv::cuda::GpuMat CMTTracker::doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream)
{
    return img;
}

void TLDTracker::Init(bool firstInit)
{

}

cv::cuda::GpuMat TLDTracker::doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream)
{
    return img;
}
