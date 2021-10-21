#include <ct/types/opencv.hpp>

#include <MetaObject/core/metaobject_config.hpp>

#include "Aquila/framegrabbers/GrabberInfo.hpp"
#include "MetaObject/params/detail/TParamPtrImpl.hpp"
#include "aqframegrabbers_link_libs.hpp"
#include "cv_capture.h"
#include "precompiled.hpp"

#ifdef _MSC_VER
RUNTIME_COMPILER_LINKLIBRARY("ole32.lib")
#endif

namespace aqframegrabbers
{
    template <class T>
    void SafeRelease(T** ppT)
    {
        if (*ppT)
        {
            (*ppT)->Release();
            *ppT = NULL;
        }
    }

    bool GrabberCV::loadData(const std::string& file_path)
    {
        if (loadGPU(file_path))
        {
            return true;
        }
        else
        {
            return loadCPU(file_path);
        }
        return false;
    }

    bool GrabberCV::loadGPU(const std::string& file_path)
    {
#if MO_OPENCV_HAVE_CUDA
        d_cam.release();
        try
        {
            auto d_temp = cv::cudacodec::createVideoReader(file_path);
            if (d_temp)
            {
                d_cam = d_temp;
                loaded_document = file_path;
                return true;
            }
        }
        catch (cv::Exception& /*e*/)
        {
        }
#else
        (void)file_path;
#endif
        return false;
    }

    bool GrabberCV::loadCPU(const std::string& file_path)
    {
        h_cam.release();
        getLogger().info("[{}::h_loadFile] Trying to load: {}", GetTypeName(), file_path);
        try
        {
            h_cam.reset(new cv::VideoCapture());
            if (h_cam)
            {
                int index = -1;
#ifdef BOOST_LEXICAL_CAST_TRY_LEXICAL_CONVERT_HPP
                if (!boost::conversion::detail::try_lexical_convert(file_path, index))
                {
                    index = -1;
                }
#else
                try
                {
                    index = boost::lexical_cast<int>(file_path);
                }
                catch (...)
                {
                    index = -1;
                }
#endif

                if (index == -1)
                {
                    if (h_cam->open(file_path))
                    {
                        loaded_document = file_path;
                        return true;
                    }
                }
                else
                {
                    if (h_cam->open(index))
                    {
                        loaded_document = file_path;
                        initial_time = mo::Time::now();
                        return true;
                    }
                }
            }
        }
        catch (cv::Exception& e)
        {
            getLogger().debug("Unable to load {} due to {}", file_path, e.what());
        }
        return false;
    }

    bool GrabberCV::grab()
    {
#if MO_OPENCV_HAVE_CUDA
        if (d_cam)
        {
            cv::cuda::GpuMat img;
            if (d_cam->nextFrame(img))
            {
                image_param.updateData(img);
                return true;
            }
        }
        else
#endif
            if (h_cam)
        {
            cv::Mat img;
            if (h_cam->read(img))
            {
                int fn = -1;
                if (query_frame_number)
                {
                    fn = static_cast<int>(h_cam->get(cv::CAP_PROP_POS_FRAMES));
                    if (fn == -1)
                    {
                        query_frame_number = false;
                    }
                }
                double ts_ = -1.0;
                if (query_time)
                {
                    ts_ = h_cam->get(cv::CAP_PROP_POS_MSEC);
                }
                mo::Time ts;
                if (ts_ < 0.0)
                {
                    if (!initial_time)
                    {
                        initial_time = mo::Time::now();
                    }
                    ts = mo::Time(mo::Time::now() - *initial_time);
                    query_time = false;
                }
                else
                {
                    ts = mo::Time(ts_ * mo::ms);
                }
                if (use_system_time)
                {
                    ts = mo::Time::now();
                }
                if (fn == -1)
                {
                    output.publish(img, mo::tags::timestamp = ts);
                }
                else
                {
                    output.publish(img, mo::tags::timestamp = ts, mo::tags::fn = static_cast<size_t>(fn));
                }
                return true;
            }
        }
        return false;
    }

    class GrabberCamera : public GrabberCV
    {
      public:
        static void listPaths(std::vector<std::string>& paths);
        static int canLoad(const std::string& doc);
        static int loadTimeout() { return 5000; }

        MO_DERIVE(GrabberCamera, GrabberCV)
            PARAM(int, width, 640)
            PARAM(int, height, 480)
            PARAM_UPDATE_SLOT(width)
            PARAM_UPDATE_SLOT(height)
            PARAM(float, focus, -1.0f)
            PARAM_UPDATE_SLOT(focus)
            PARAM(float, exposure, -1.0f)
            PARAM_UPDATE_SLOT(exposure)
        MO_END;

        bool loadData(const std::string& path) override;

      protected:
        bool processImpl() override;
    };

    void GrabberCamera::on_exposure_modified(const mo::IParam&, mo::Header, mo::UpdateFlags, mo::IAsyncStream&)
    {
        mo::Mutex_t::Lock_t lock(getMutex());
        if (h_cam)
        {
            if (exposure < 0)
            {
                h_cam->set(cv::CAP_PROP_AUTO_EXPOSURE, 1);
            }
            else
            {
                h_cam->set(cv::CAP_PROP_AUTO_EXPOSURE, 0);
                h_cam->set(cv::CAP_PROP_EXPOSURE, exposure);
            }
        }
    }

    void GrabberCamera::on_focus_modified(const mo::IParam&, mo::Header, mo::UpdateFlags, mo::IAsyncStream&)
    {
        mo::Mutex_t::Lock_t lock(getMutex());
        if (h_cam)
        {
            if (focus > 0.0f)
            {
                h_cam->set(cv::CAP_PROP_AUTOFOCUS, 0);
                if (!h_cam->set(cv::CAP_PROP_FOCUS, focus))
                {
                    this->getLogger().debug("Failed to set focus on device");
                }
            }
            else
            {
                h_cam->set(cv::CAP_PROP_AUTOFOCUS, 1);
            }
        }
    }

    void GrabberCamera::on_height_modified(const mo::IParam&, mo::Header, mo::UpdateFlags, mo::IAsyncStream&)
    {
        mo::Mutex_t::Lock_t lock(getMutex());
        if (h_cam)
        {
            h_cam->set(cv::CAP_PROP_FRAME_HEIGHT, height);
        }
    }

    void GrabberCamera::on_width_modified(const mo::IParam&, mo::Header, mo::UpdateFlags, mo::IAsyncStream&)
    {
        mo::Mutex_t::Lock_t lock(getMutex());
        if (h_cam)
        {
            h_cam->set(cv::CAP_PROP_FRAME_WIDTH, width);
        }
    }

    bool GrabberCamera::processImpl()
    {
        if (h_cam)
        {
            if (focus_param.getModified())
            {
                if (focus > 0)
                {
                    h_cam->set(cv::CAP_PROP_AUTOFOCUS, 0);
                    h_cam->set(cv::CAP_PROP_FOCUS, focus);
                }
                else
                {
                    h_cam->set(cv::CAP_PROP_AUTOFOCUS, 1);
                }
                focus_param.setModified(false);
            }
        }
        return GrabberCV::processImpl();
    }

    void GrabberCamera::listPaths(std::vector<std::string>& paths)
    {
#ifdef _MSC_VER
        MFStartup(MF_VERSION);
        HRESULT hr = S_OK;
        IMFAttributes* pAttributes = NULL;
        UINT32 m_cDevices;                // contains the number of devices
        IMFActivate** m_ppDevices = NULL; // contains properties about each device

        // Initialize an attribute store. We will use this to
        // specify the enumeration parameters.

        hr = MFCreateAttributes(&pAttributes, 1);

        // Ask for source type = video capture devices
        if (SUCCEEDED(hr))
        {
            hr = pAttributes->SetGUID(MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE,
                                      MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_GUID);
        }
        // Enumerate devices.
        if (SUCCEEDED(hr))
        {
            hr = MFEnumDeviceSources(pAttributes, &m_ppDevices, &m_cDevices);
        }
        for (uint32_t i = 0; i < m_cDevices; ++i)
        {
            HRESULT hr = S_OK;
            wchar_t* ppszName = nullptr;
            hr = m_ppDevices[i]->GetAllocatedString(MF_DEVSOURCE_ATTRIBUTE_FRIENDLY_NAME, &ppszName, NULL);
            std::wstring wstring(ppszName);
            paths.push_back(boost::lexical_cast<std::string>(i) + " - " + std::string(wstring.begin(), wstring.end()));
        }

        SafeRelease(&pAttributes);

        for (UINT32 i = 0; i < m_cDevices; i++)
        {
            SafeRelease(&m_ppDevices[i]);
        }
        CoTaskMemFree(m_ppDevices);
        m_ppDevices = NULL;

        m_cDevices = 0;
#else
        (void)paths;
#endif
    }

    int GrabberCamera::canLoad(const std::string& doc)
    {
        auto pos = doc.find(" - ");
#ifdef BOOST_LEXICAL_CAST_TRY_LEXICAL_CONVERT_HPP
        if (pos != std::string::npos)
        {
            int index = 0;
            if (boost::conversion::detail::try_lexical_convert(doc.substr(0, pos), index))
            {
                return 10;
            }
        }
        else
        {
            int index = 0;
            if (boost::conversion::detail::try_lexical_convert(doc, index))
            {
                return 10;
            }
        }
#else
        if (pos != std::string::npos)
        {
            try
            {
                int index = boost::lexical_cast<int>(doc.substr(0, pos));
            }
            catch (...)
            {
            }
        }
        else
        {
            try
            {
                int index = boost::lexical_cast<int>(doc);
                return 10;
            }
            catch (...)
            {
            }
        }
#endif
        std::vector<std::string> cameras;
        listPaths(cameras);
        for (const auto& camera : cameras)
        {
            if (camera == doc)
                return 10;
        }
        return 0;
    }

    bool GrabberCamera::loadData(const std::string& file_path)
    {
        int index = 0;
#ifdef BOOST_LEXICAL_CAST_TRY_LEXICAL_CONVERT_HPP
        if (boost::conversion::detail::try_lexical_convert(file_path, index))
        {
            h_cam.reset(new cv::VideoCapture(index));
            initial_time = mo::Time::now();
            return true;
        }
        else
        {
            index = 0;
        }
#else
        try
        {
            index = boost::lexical_cast<int>(file_path);
            h_cam.reset(new cv::VideoCapture(index));
            initial_time = mo::getCurrentTime();
            return true;
        }
        catch (...)
        {
            index = 0;
        }
#endif
        std::vector<std::string> cameras;
        listPaths(cameras);
        for (size_t i = 0; i < cameras.size(); ++i)
        {
            if (cameras[i] == file_path)
            {
                h_cam.reset(new cv::VideoCapture());

                if (h_cam->open(i))
                {
                    initial_time = mo::Time::now();
                    return true;
                }
            }
            ++index;
        }
        auto func = [&cameras]() {
            std::stringstream ss;
            for (auto& cam : cameras)
                ss << cam << ", ";
            return ss.str();
        };
        this->getLogger().debug("Unable to load {} queried cameras: trying to requery", file_path, func());

        listPaths(cameras);
        for (size_t i = 0; i < cameras.size(); ++i)
        {
            if (cameras[i] == file_path)
            {
                h_cam.reset(new cv::VideoCapture());
                if (h_cam->open(i))
                {
                    initial_time = mo::Time::now();
                    return true;
                }
            }
            ++index;
        }
        this->getLogger().warn("Unable to load {}  queried cameras: {}", file_path, func());
        return false;
    }
} // namespace aqframegrabbers
using namespace aqframegrabbers;
MO_REGISTER_CLASS(GrabberCamera)
