#include "AxisCamera.h"
#include "QtNetwork/qauthenticator.h"

using namespace EagleLib;

AxisSocket::AxisSocket()
{
    network_manager = new QNetworkAccessManager(this);
    network_request = nullptr;
    connect(network_manager, SIGNAL(authenticationRequired(QNetworkReply*,QAuthenticator*)), this, SLOT(onAuthenticationRequired(QNetworkReply*,QAuthenticator*)));
}

void AxisSocket::request(QUrl url)
{
    network_request = network_manager->get(QNetworkRequest(url));
    connect(network_request, SIGNAL(finished()), this, SLOT(requestFinished()));
    connect(network_request, SIGNAL(readyRead()), this, SLOT(requestReadyRead()));
    connect(network_request, SIGNAL(downloadProgress(quint64,quint64)), this, SLOT(requestDownloadProgress(quint64,quint64)));
}

void AxisSocket::requestReadyRead()
{
    qDebug() << "Request ready to read";
    qDebug() << network_request->readAll();
}

void AxisSocket::requestFinished()
{

}

void AxisSocket::requestDownloadProgress(quint64 received, quint64 total)
{

}

void AxisSocket::onAuthenticationRequired(QNetworkReply* reply, QAuthenticator* auth)
{
    qDebug() << "Authentication required";
	auth->setUser(username);
	auth->setPassword(password);
}

void AxisCamera::on_panRequest()
{
    int port = *getParameter<int>("Camera port")->Data();
    int tilt = *getParameter<int>("Camera pan")->Data();
    int cameraNum = *getParameter<int>("Camera number")->Data();
	std::string portStr;
	if (port != -1)
		portStr = boost::lexical_cast<std::string>(port);
	std::string tiltStr = (tilt < 0) ? "lpan=" : "rpan=";
	std::string requestStr = "http://" + *getParameter<std::string>("Camera address")->Data() +
		portStr + "/axis-cgi/ptz.cgi?" +
		tiltStr + boost::lexical_cast<std::string>(abs(tilt)) +
		"&camera=" + boost::lexical_cast<std::string>(cameraNum);
    log(Status, "Pan request: " + requestStr);
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

    log(Status, "Zoom request: " + requestStr);
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

    log(Status, "Tilt request: " + requestStr);
    socket->request(QUrl(QString::fromStdString(requestStr)));
    updateParameter("Camera moving", true);
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
void AxisCamera::Init(bool firstInit)
{
	
	if (firstInit) 
	{
		socket.reset(new AxisSocket());
        updateParameter<std::string>("Camera address", "192.168.0.6");
		updateParameter("Camera port", int(-1));
        updateParameter<std::string>("Camera username", "root"); 
        updateParameter<std::string>("Camera password", "12369pp"); // TODO change to password string

		updateParameter("Camera calibration file", Parameters::Parameter::ReadFile("Camera_Calibration.yml"));
		
		updateParameter("Camera zoom", int(0));
		updateParameter("Camera pan", int(0));
		updateParameter("Camera tilt", int(0));
        updateParameter("Camera number", int(1));

        updateParameter("Camera moving", false, Parameters::Parameter::State);

        updateParameter("Camera matrix", cv::Mat(), Parameters::Parameter::Output);
		updateParameter("Camera pose", cv::Mat(), Parameters::Parameter::Output);
	}
    zoomConnection = getParameter("Camera zoom")->RegisterNotifier(boost::bind(&AxisCamera::on_zoomRequest, this));
    panConnection = getParameter("Camera pan")->RegisterNotifier(boost::bind(&AxisCamera::on_panRequest, this));
    tiltConnection = getParameter("Camera tilt")->RegisterNotifier(boost::bind(&AxisCamera::on_tiltRequest, this));
	ipConnection = getParameter("Camera address")->RegisterNotifier(boost::bind(&AxisCamera::on_addressChange, this));
	usernameConnection = getParameter("Camera username")->RegisterNotifier(boost::bind(&AxisCamera::on_credentialChange, this));
	passwordConnection = getParameter("Camera password")->RegisterNotifier(boost::bind(&AxisCamera::on_credentialChange, this));

}

cv::cuda::GpuMat AxisCamera::doProcess(cv::cuda::GpuMat& img, cv::cuda::Stream& stream)
{

	return img;
}
bool AxisCamera::SkipEmpty() const
{
	return false;
}

NODE_DEFAULT_CONSTRUCTOR_IMPL(AxisCamera)
ADD_RUNTIME_SOURCE_DEPENDENCY_ABS("E:/code/build/EagleEye/Plugins/Vclick3d/src/moc_AxisCamera.cpp")