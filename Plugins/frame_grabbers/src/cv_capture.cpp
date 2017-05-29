#include "cv_capture.h"
#include "precompiled.hpp"
#include "Aquila/framegrabbers/GrabberInfo.hpp"
#include "MetaObject/params/detail/TParamPtrImpl.hpp"
#if _MSC_VER
RUNTIME_COMPILER_LINKLIBRARY("ole32.lib")
#endif
using namespace aq;
using namespace aq::Nodes;




template <class T> void SafeRelease(T **ppT)
{
    if (*ppT)
    {
        (*ppT)->Release();
        *ppT = NULL;
    }
}


bool GrabberCV::loadData(const std::string& file_path)
{
    if(LoadGPU(file_path))
    {
        return true;
    }else
    {
        return LoadCPU(file_path);
    }
    return false;
}

bool GrabberCV::LoadGPU(const std::string& file_path)
{
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
    catch (cv::Exception& e)
    {

    }
    return false;
}

bool GrabberCV::LoadCPU(const std::string& file_path)
{
    h_cam.release();
    //LOG(info) << "Attemping to load " << file_path;
    BOOST_LOG_TRIVIAL(info ) << "[" << GetTypeName() << "::h_loadFile] Trying to load: \"" << file_path << "\"";
    try
    {
        h_cam.reset(new cv::VideoCapture());
        if (h_cam)
        {
            int index = -1;
            if(!boost::conversion::detail::try_lexical_convert(file_path, index))
            {
                index = -1;
            }

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
                    initial_time = boost::posix_time::microsec_clock::universal_time();
                    return true;
                }
            }
        }
    }
    catch (cv::Exception& e)
    {
        LOG(debug) << "Unable to load " << file_path << " due to " << e.what();
    }
    return false;
}

bool GrabberCV::grab()
{
    if(d_cam)
    {
        cv::cuda::GpuMat img;
        if(d_cam->nextFrame(img))
        {
            image_param.updateData(img);
            return true;
        }
    }else if(h_cam)
    {
        cv::Mat img;
        if(h_cam->read(img))
        {
            double fn = h_cam->get(CV_CAP_PROP_POS_FRAMES);
            double ts_ = h_cam->get(CV_CAP_PROP_POS_MSEC);
            mo::Time_t ts;
            
            if(ts_ == -1)
            {
                ts = mo::Time_t((boost::posix_time::microsec_clock::universal_time() - initial_time).total_microseconds() * mo::ms);
            }else
            {
                ts = mo::Time_t(ts_* mo::ms);
            }
            if(fn == -1)
            {
                image_param.updateData(img, mo::tag::_timestamp = ts, _ctx.get());
            }else
            {
                image_param.updateData(img, mo::tag::_timestamp = ts, mo::tag::_frame_number = fn, _ctx.get());
            }
            return true;
        }
    }
    return false;
}

class GrabberCamera:public GrabberCV
{
public:
    MO_DERIVE(GrabberCamera, GrabberCV)

    MO_END;
    bool loadData(const std::string& path);
    static void ListPaths(std::vector<std::string>& paths);
    static int CanLoad(const std::string& doc);
    static int Timeout()
    {
        return 5000;
    }
};  

void GrabberCamera::ListPaths(std::vector<std::string>& paths)
{
#ifdef _MSC_VER
    MFStartup(MF_VERSION);
    HRESULT hr = S_OK;
    IMFAttributes *pAttributes = NULL;
    UINT32      m_cDevices; // contains the number of devices
    IMFActivate **m_ppDevices = NULL; // contains properties about each device

                                      // Initialize an attribute store. We will use this to
                                      // specify the enumeration parameters.

    hr = MFCreateAttributes(&pAttributes, 1);

    // Ask for source type = video capture devices
    if (SUCCEEDED(hr))
    {
        hr = pAttributes->SetGUID(
            MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE,
            MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_GUID
        );
    }
    // Enumerate devices.
    if (SUCCEEDED(hr))
    {
        hr = MFEnumDeviceSources(pAttributes, &m_ppDevices, &m_cDevices);
    }
    for (int i = 0; i < m_cDevices; ++i)
    {
        HRESULT hr = S_OK;
        wchar_t* ppszName = nullptr;
        hr = m_ppDevices[i]->GetAllocatedString(
            MF_DEVSOURCE_ATTRIBUTE_FRIENDLY_NAME,
            &ppszName,
            NULL
        );
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



#endif
}

int GrabberCamera::CanLoad(const std::string& doc)
{
    auto pos = doc.find(" - ");
    if (pos != std::string::npos)
    {
        int index = 0;
        if (boost::conversion::detail::try_lexical_convert(doc.substr(pos), index))
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
    std::vector<std::string> cameras;
    ListPaths(cameras);
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
    if (boost::conversion::detail::try_lexical_convert(file_path, index))
    {
        h_cam.reset(new cv::VideoCapture(index));
        initial_time = boost::posix_time::microsec_clock::universal_time();
        return true;
    }
    else
    {
        index = 0;
    }
    std::vector<std::string> cameras;
    ListPaths(cameras);
    for(int i = 0; i < cameras.size(); ++i)
    {
        if (cameras[i] == file_path)
        {
            h_cam.reset(new cv::VideoCapture());
            
            if(h_cam->open(i))
            {
                initial_time = boost::posix_time::microsec_clock::universal_time();
                return true;
            }
        }
        ++index;
    }
    auto func = [&cameras]()
    {
        std::stringstream ss;
        for (auto& cam : cameras)
            ss << cam << ", ";
        return ss.str();
    };
    LOG(debug) << "Unable to load " << file_path << " queried cameras: " << func() << " trying to requery";

    ListPaths(cameras);
    for(int i = 0; i < cameras.size(); ++i)
    {
        if (cameras[i] == file_path)
        {
            h_cam.reset(new cv::VideoCapture());
            if(h_cam->open(i))
            {
                initial_time = boost::posix_time::microsec_clock::universal_time();
                return true;
            }
        }
        ++index;
    }
    LOG(warning) << "Unable to load " << file_path << " queried cameras: " << func();
    return false;
}

MO_REGISTER_CLASS(GrabberCamera)
