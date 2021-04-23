#define NOMINMAX
#include "frame_grabber_openni2.h"
#include "openni2_initializer.h"
#include <Aquila/framegrabbers/FrameGrabberInfo.hpp>
#include <MetaObject/logging/profiling.hpp>
#include <opencv2/core.hpp>

namespace aqopenni2
{

    int frame_grabber_openni2::canLoadPath(const std::string& document)
    {
        std::string doc = document;
        std::transform(doc.begin(), doc.end(), doc.begin(), ::tolower);
        std::string openni("openni::");
        if (doc.compare(0, openni.length(), openni) == 0)
        {
            // Check if valid uri
            aq::initializer_NI2::instance();
            openni::Array<openni::DeviceInfo> devices;
            openni::OpenNI::enumerateDevices(&devices);
            for (int i = 0; i < devices.getSize(); ++i)
            {
                auto uri = devices[i].getUri();
                auto len = strlen(uri);
                if (document.compare(openni.length(), len, uri) == 0)
                {
                    return 11;
                }
            }
            return 5;
        }
        return 0;
    }

    int frame_grabber_openni2::loadTimeout() { return 10000; }

    std::vector<std::string> frame_grabber_openni2::listLoadablePaths()
    {
        aq::initializer_NI2::instance();
        openni::Array<openni::DeviceInfo> devices;
        openni::OpenNI::enumerateDevices(&devices);

        std::vector<std::string> output;
        for (int i = 0; i < devices.getSize(); ++i)
        {
            auto uri = devices[i].getUri();
            output.push_back(std::string("OpenNI::") + std::string(uri));
        }
        return output;
    }

    frame_grabber_openni2::~frame_grabber_openni2()
    {
        if (_depth)
        {
            _depth->removeNewFrameListener(this);
            _depth->stop();
            if (_device)
            {
                _device->close();
            }
        }
    }

    bool frame_grabber_openni2::loadData(std::string file_path)
    {
        std::string doc = file_path;
        std::transform(doc.begin(), doc.end(), doc.begin(), ::tolower);
        std::string openni("openni::");
        if (doc.compare(0, openni.length(), openni) == 0 || file_path.empty())
        {
            aq::initializer_NI2::instance();
            std::string uri;
            if (!file_path.empty())
            {
                uri = file_path.substr(openni.length());
            }
            _device.reset(new openni::Device());

            openni::Status rc;
            if (uri.size())
                rc = _device->open(uri.c_str());
            else
                rc = _device->open(openni::ANY_DEVICE);
            if (rc != openni::STATUS_OK)
            {
                MO_LOG(info, "Unable to connect to openni2 compatible device: {}", openni::OpenNI::getExtendedError());
                return false;
            }
            _depth.reset(new openni::VideoStream());
            _color = std::make_shared<openni::VideoStream>();

            rc = _depth->create(*_device, openni::SENSOR_DEPTH);
            if (rc != openni::STATUS_OK)
            {
                MO_LOG(info, "Unable to retrieve depth stream: {}", openni::OpenNI::getExtendedError());
                return false;
            }
            rc = _color->create(*_device, openni::SENSOR_COLOR);
            if (rc != openni::STATUS_OK)
            {
                MO_LOG(info, "Unable to retrieve color stream: {}", openni::OpenNI::getExtendedError());
                return false;
            }
            // openni::VideoMode mode = _depth->getVideoMode();
            // mode.setResolution(640, 480);
            // rc = _depth->setVideoMode(mode);
            if (rc != openni::STATUS_OK)
            {
                MO_LOG(info, "Unable to set video resolution");
            }
            _color->addNewFrameListener(this);
            _color->start();
            _depth->addNewFrameListener(this);
            _depth->start();
            MO_LOG(info, "Connected to device ", _device->getDeviceInfo().getUri());
            return true;
        }
        return false;
    }

    void frame_grabber_openni2::onNewFrame(openni::VideoStream& stream)
    {
        if (&stream == _depth.get())
        {
            openni::Status rc = stream.readFrame(&_frame);
            if (rc != openni::STATUS_OK)
            {
                MO_LOG(debug, "Unable to read new depth frame: ", openni::OpenNI::getExtendedError());
                return;
            }
            int height = _frame.getHeight();
            int width = _frame.getWidth();
            // auto ts = _frame.getTimestamp();
            auto fn = _frame.getFrameIndex();
            depth_fn = fn;
            // int scale = 1;
            switch (_frame.getVideoMode().getPixelFormat())
            {
            case openni::PIXEL_FORMAT_DEPTH_100_UM: {
                // scale = 10;
            }

            case openni::PIXEL_FORMAT_DEPTH_1_MM: {
                cv::Mat data(height, width, CV_16U, (ushort*)_frame.getData());
                cv::Mat XYZ;
                XYZ.create(height, width, CV_32FC3);
                for (int i = 0; i < height; ++i)
                {
                    ushort* ptr = data.ptr<ushort>(i);
                    cv::Vec3f* pt_ptr = XYZ.ptr<cv::Vec3f>(i);
                    for (int j = 0; j < width; ++j)
                    {
                        openni::CoordinateConverter::convertDepthToWorld(
                            *_depth, j, i, ptr[j], &pt_ptr[j].val[0], &pt_ptr[j].val[1], &pt_ptr[j].val[2]);
                    }
                }
                mo::Mutex_t::Lock_t lock(getMutex());
                new_xyz = XYZ;
                new_depth = data.clone();
                INode* node = this;
                sig_node_updated(node);
                break;
            }
            default:
                break;
            }
            return;
        }
        openni::Status rc = stream.readFrame(&_color_frame);
        if (rc != openni::STATUS_OK)
        {
            MO_LOG(debug, "Unable to read new depth frame: ", openni::OpenNI::getExtendedError());
            return;
        }
        int height = _color_frame.getHeight();
        int width = _color_frame.getWidth();
        // auto ts = _frame.getTimestamp();
        auto fn = _color_frame.getFrameIndex();
        int scale = 1;
        auto pixel_format = _color_frame.getVideoMode().getPixelFormat();
        color_fn = fn;
        switch (pixel_format)
        {
        default:
            break;
        case openni::PIXEL_FORMAT_RGB888: {
            cv::Mat data(height, width, CV_8UC3, (ushort*)_color_frame.getData());
            mo::Mutex_t::Lock_t lock(getMutex());
            new_color = data.clone();
            INode* node = this;
            sig_node_updated(node);
        }
        }
    }

    bool frame_grabber_openni2::processImpl()
    {
        if (!new_xyz.empty() && !new_depth.empty())
        {
            xyz.publish(new_xyz, mo::tags::timestamp = mo::Time::now(), mo::tags::fn = depth_fn);
            depth.publish(new_depth, mo::tags::timestamp = mo::Time::now(), mo::tags::fn = depth_fn);
            new_xyz.release();
            new_depth.release();
        }
        if (!new_color.empty())
        {
            color.publish(new_color, mo::tags::timestamp = mo::Time::now(), mo::tags::fn = color_fn);
            new_color.release();
        }

        return true;
    }
} // namespace aqopenni2
using namespace aqopenni2;

MO_REGISTER_CLASS(frame_grabber_openni2);
