#include "AxisCamera.h"
#include "QtNetwork/qauthenticator.h"

using namespace EagleLib;

SETUP_PROJECT_IMPL

/*
void SetupIncludes()
{
#ifdef PROJECT_INCLUDES
	{
		std::string str(PROJECT_INCLUDES);
		std::string::size_type pos = str.find_first_of('+');
		std::string::size_type prevPos = 0;
		while (pos != std::string::npos)
		{
			EagleLib::NodeManager::getInstance().addIncludeDir(str.substr(prevPos, pos - prevPos).c_str());
			prevPos = pos + 1;
			pos = str.find_first_of('+', pos + 1);
		}
	}
#endif 
#ifdef PROJECT_LIB_DIRS
	{
		std::string str(PROJECT_LIB_DIRS);
		std::string::size_type pos = str.find_first_of('+');
		std::string::size_type prevPos = 0;
		while (pos != std::string::npos)
		{
			EagleLib::NodeManager::getInstance().addLinkDir(str.substr(prevPos, pos - prevPos).c_str());
			prevPos = pos + 1;
			pos = str.find_first_of('+', pos + 1);  
		}
	}
#endif
}*/



 
AxisSocket::AxisSocket() 
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
void AxisCamera::Init(bool firstInit)
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

        updateParameter("Camera moving", false, Parameters::Parameter::State);

		currentZoom.reset(new Parameters::TypedParameter<int>("Current Zoom", 0, Parameters::Parameter::State));
		currentFocus.reset(new Parameters::TypedParameter<int>("Current Focus", 0, Parameters::Parameter::State));
		socket->zoom = currentZoom;
		socket->focus = currentFocus;
		parameters.push_back(currentZoom);
		parameters.push_back(currentFocus);

		updateParameter("Camera matrix", cv::Mat(), Parameters::Parameter::Output);
		updateParameter("Camera pose", cv::Mat(), Parameters::Parameter::Output);
	}
	else
	{
		currentZoom = std::dynamic_pointer_cast<Parameters::TypedParameter<int>>(getParameter<int>("Current Zoom"));
		currentFocus = std::dynamic_pointer_cast<Parameters::TypedParameter<int>>(getParameter<int>("Current Focus"));
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
	zoomConnection.disconnect(); 
	panConnection.disconnect();
	tiltConnection.disconnect();
	ipConnection.disconnect();
	usernameConnection.disconnect();
	passwordConnection.disconnect();
}

cv::cuda::GpuMat AxisCamera::doProcess(cv::cuda::GpuMat& img, cv::cuda::Stream& stream)
{
	return img;
}
bool AxisCamera::SkipEmpty() const 
{
	return false;
}
ADD_RUNTIME_SOURCE_DEPENDENCY_ABS("E:/build/EagleEye/Plugins/Vclick3d/src/moc_AxisCamera.cpp")
ADD_RUNTIME_SOURCE_DEPENDENCY_ABS("E:/build/EagleEye/Plugins/Vclick3d/include/moc_AxisCamera.cpp")
NODE_DEFAULT_CONSTRUCTOR_IMPL(AxisCamera)
