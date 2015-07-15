#pragma once
#include "nodes/Node.h"
#include "Manager.h"
#include "QtNetwork/qnetworkaccessmanager.h"
#include "QtNetwork/QNetworkReply.h"
#include "QtNetwork/qtcpsocket.h"
#include "ObjectInterfacePerModule.h"

#ifdef _DEBUG
RUNTIME_COMPILER_LINKLIBRARY("Qt5Cored.lib");
RUNTIME_COMPILER_LINKLIBRARY("Qt5Networkd.lib");
RUNTIME_COMPILER_LINKLIBRARY("Qt5Guid.lib");
RUNTIME_COMPILER_LINKLIBRARY("Qt5Widgetsd.lib");
#else
RUNTIME_COMPILER_LINKLIBRARY("Qt5Core.lib");
RUNTIME_COMPILER_LINKLIBRARY("Qt5Network.lib");
RUNTIME_COMPILER_LINKLIBRARY("Qt5Gui.lib");
RUNTIME_COMPILER_LINKLIBRARY("Qt5Widgets.lib");
#endif

SETUP_PROJECT_DEF


namespace EagleLib
{
	class AxisSocket: public QObject
	{
		Q_OBJECT
		QNetworkAccessManager* network_manager;
        QNetworkReply* network_request;
		
	public:
		Parameters::ITypedParameter<int>::Ptr zoom;
		Parameters::ITypedParameter<int>::Ptr focus;

		AxisSocket();
		void request(QUrl url);
		QString username;
		QString password;
    private slots:
        void requestReadyRead();
        void requestFinished();
        void requestDownloadProgress(quint64 received, quint64 total);
		void onAuthenticationRequired(QNetworkReply* reply, QAuthenticator* auth);
		

	};


	class AxisCamera : public Node
	{
        boost::signals2::connection zoomConnection;
        boost::signals2::connection panConnection;
        boost::signals2::connection tiltConnection;
		boost::signals2::connection ipConnection;
		boost::signals2::connection usernameConnection;
		boost::signals2::connection passwordConnection;
        boost::shared_ptr<AxisSocket> socket;
		
        void on_zoomRequest();
        void on_panRequest();
        void on_tiltRequest();
		void on_addressChange();
		void on_credentialChange();
		void get_position();
		Parameters::TypedParameter<int>::Ptr currentZoom;
		Parameters::TypedParameter<int>::Ptr currentFocus;

	public:
		AxisCamera();
		~AxisCamera();
		virtual void Init(bool firstInit);
		virtual void Serialize(ISimpleSerializer* pSerializer);
		virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat& img, cv::cuda::Stream& stream = cv::cuda::Stream::Null());
		virtual bool SkipEmpty() const;

	};
}
