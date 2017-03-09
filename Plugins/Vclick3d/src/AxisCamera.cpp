#include "AxisCamera.h"
#include "QtNetwork/qauthenticator.h"


#include <boost/algorithm/string/predicate.hpp>
#include <boost/lexical_cast.hpp>

using namespace aq;
using namespace aq::Nodes;

SETUP_PROJECT_IMPL

 
/*AxisSocket::AxisSocket() 
{
    network_manager = new QNetworkAccessManager(this);
    network_request = nullptr;
    connect(network_manager, SIGNAL(authenticationRequired(QNetworkReply*,QAuthenticator*)), this, SLOT(onAuthenticationRequired(QNetworkReply*,QAuthenticator*)));
}

void AxisSocket::request(QUrl url)
{
    qDebug() << "Sending Request: " << url; 
    network_request = network_manager->get(QNetworkRequest(url));
    connect(network_request, SIGNAL(finished()), this, SLOT(requestFinished()));
    connect(network_request, SIGNAL(readyRead()), this, SLOT(requestReadyRead()));
    connect(network_request, SIGNAL(downloadProgress(quint64,quint64)), this, SLOT(requestDownloadProgress(quint64,quint64)));
} 

void AxisSocket::requestReadyRead()
{
    qDebug() << "Request ready to read";
    QByteArray line = network_request->readLine();
    
    while (line.size())
    {
        std::string str = QString(line).toStdString();
        int idx = str.find('=');
        std::string first = str.substr(0, idx);
        std::string second = str.substr(idx + 1, str.size() - 3 - first.size());
        std::cout << first << " " << second << std::endl;
        if (first == "zoom")
        {
            zoom->UpdateData(boost::lexical_cast<int>(second));
        }
        if (first == "focus")
        {
            focus->UpdateData(boost::lexical_cast<int>(second));
        }
        line = network_request->readLine();
    }
    
}

void AxisSocket::requestFinished()
{

}

void AxisSocket::requestDownloadProgress(quint64 received, quint64 total)
{

}

void AxisSocket::onAuthenticationRequired(QNetworkReply* reply, QAuthenticator* auth)
{
    //qDebug() << "Authentication required";
    auth->setUser(username);
    auth->setPassword(password);
}

void AxisCamera::on_panRequest()
{
    int port = *getParameter<int>("Camera port")->Data();
    int tilt = *getParameter<int>("Camera pan")->Data();
    std::string userName = *getParameter<std::string>("Camera username")->Data();
    std::string pass = *getParameter<std::string>("Camera password")->Data();
    int cameraNum = *getParameter<int>("Camera number")->Data();
    std::string portStr;
    if (port != -1)
        portStr = boost::lexical_cast<std::string>(port);
    std::string tiltStr = (tilt < 0) ? "lpan=" : "rpan=";
    std::string requestStr = "http://";
    if (userName.size() && pass.size())
        requestStr += userName + ":" + pass + "@";
    requestStr += *getParameter<std::string>("Camera address")->Data() +
        portStr + "/axis-cgi/ptz.cgi?" +
        tiltStr + boost::lexical_cast<std::string>(abs(tilt)) +
        "&camera=" + boost::lexical_cast<std::string>(cameraNum);

    NODE_LOG(info) << "Pan request: " << requestStr;
    socket->request(QUrl(QString::fromStdString(requestStr)));
    updateParameter("Camera moving", true);
}
void AxisCamera::on_zoomRequest()
{ 
    int port = *getParameter<int>("Camera port")->Data();
    int tilt = *getParameter<int>("Camera zoom")->Data();
    int cameraNum = *getParameter<int>("Camera number")->Data();
    std::string portStr;
    if (port != -1)
        portStr = boost::lexical_cast<std::string>(port);
    std::string tiltStr = "zoom=";
    std::string requestStr = "http://" + *getParameter<std::string>("Camera address")->Data() +
        portStr + "/axis-cgi/com/ptz.cgi?" +
        tiltStr + boost::lexical_cast<std::string>(abs(tilt)) +
        "&camera=" + boost::lexical_cast<std::string>(cameraNum);

    NODE_LOG(info) << "Zoom request: " << requestStr;
    socket->request(QUrl(QString::fromStdString(requestStr)));
    updateParameter("Camera moving", true);
}
void AxisCamera::on_tiltRequest() 
{
    int port = *getParameter<int>("Camera port")->Data();
    int tilt = *getParameter<int>("Camera tilt")->Data();
    int cameraNum = *getParameter<int>("Camera number")->Data();
    std::string portStr;
    if (port != -1)
        portStr = boost::lexical_cast<std::string>(port);
    std::string tiltStr = (tilt < 0) ? "ltilt=" : "rtilt=";
    std::string requestStr = "http://" + *getParameter<std::string>("Camera address")->Data() +
            portStr + "/axis-cgi/ptz.cgi?" +
            tiltStr + boost::lexical_cast<std::string>(abs(tilt)) + 
            "&camera=" + boost::lexical_cast<std::string>(cameraNum);

    NODE_LOG(info) << "Tilt request: " << requestStr;
    socket->request(QUrl(QString::fromStdString(requestStr)));
    updateParameter("Camera moving", true);
    on_credentialChange();
}
void AxisCamera::get_position()
{
    QString url = "http://" + QString::fromStdString(*getParameter<std::string>("Camera address")->Data()) + "/axis-cgi/com/ptz.cgi?query=position";
    socket->request(QUrl(url));
}
void AxisCamera::on_addressChange()
{

}
void AxisCamera::on_credentialChange()
{
    socket->username = QString::fromStdString(*getParameter<std::string>("Camera username")->Data());
    socket->password = QString::fromStdString(*getParameter<std::string>("Camera password")->Data());
}
void AxisCamera::Serialize(ISimpleSerializer* pSerializer)
{
    Node::Serialize(pSerializer); 
    SERIALIZE(socket);
}
void AxisCamera::NodeInit(bool firstInit)
{
    if (firstInit) 
    {
        socket.reset(new AxisSocket());

        updateParameter<std::string>("Camera address", "192.168.1.152");
        updateParameter("Camera port", int(-1));
        updateParameter<std::string>("Camera username", "root"); 
        updateParameter<std::string>("Camera password", "12369pp"); // TODO change to password string

        updateParameter("Camera calibration file", Parameters::ReadFile("Camera_Calibration.yml"));
        
        updateParameter("Camera zoom", int(0));
        updateParameter("Camera pan", int(0));
        updateParameter("Camera tilt", int(0));
        updateParameter("Camera number", int(1));

        updateParameter("Camera moving", false)->type = Parameters::Parameter::State;

        currentZoom = new Parameters::TypedParameter<int>("Current Zoom", 0, Parameters::Parameter::State);
        currentFocus = new Parameters::TypedParameter<int>("Current Focus", 0, Parameters::Parameter::State);
        socket->zoom = currentZoom;
        socket->focus = currentFocus;
        _parameters.push_back(currentZoom);
        _parameters.push_back(currentFocus);

        updateParameter("Camera matrix", cv::Mat())->type = Parameters::Parameter::Output;
        updateParameter("Camera pose", cv::Mat())->type =  Parameters::Parameter::Output;
    }
    else
    {
        currentZoom = getParameter<int>("Current Zoom");
        currentFocus = getParameter<int>("Current Focus");
    }
    updateParameter<boost::function<void(void)>>("Get position", boost::bind(&AxisCamera::get_position, this));

    zoomConnection = getParameter("Camera zoom")->RegisterNotifier(boost::bind(&AxisCamera::on_zoomRequest, this));
    panConnection = getParameter("Camera pan")->RegisterNotifier(boost::bind(&AxisCamera::on_panRequest, this));
    tiltConnection = getParameter("Camera tilt")->RegisterNotifier(boost::bind(&AxisCamera::on_tiltRequest, this));
    ipConnection = getParameter("Camera address")->RegisterNotifier(boost::bind(&AxisCamera::on_addressChange, this));
    usernameConnection = getParameter("Camera username")->RegisterNotifier(boost::bind(&AxisCamera::on_credentialChange, this));
    passwordConnection = getParameter("Camera password")->RegisterNotifier(boost::bind(&AxisCamera::on_credentialChange, this));

    on_credentialChange();
}

AxisCamera::~AxisCamera()
{
    
}

cv::cuda::GpuMat AxisCamera::doProcess(cv::cuda::GpuMat& img, cv::cuda::Stream& stream)
{
    return img;
}
bool AxisCamera::SkipEmpty() const 
{
    return false;
}




NODE_DEFAULT_CONSTRUCTOR_IMPL(AxisCamera, Utility, Network, Control)
*/

