#pragma once
#include "nodes/Node.h"
#include "Manager.h"
#include "qnetworkaccessmanager.h"
#include "qnetworkreply.h"
#include "qtcpsocket.h"
#include "ObjectInterfacePerModule.h"



#ifdef _MSC_VER
#ifdef _DEBUG
RUNTIME_COMPILER_LINKLIBRARY("Qt5Cored.lib");
RUNTIME_COMPILER_LINKLIBRARY("Qt5Networkd.lib");
RUNTIME_COMPILER_LINKLIBRARY("Qt5Guid.lib");
RUNTIME_COMPILER_LINKLIBRARY("Qt5Widgetsd.lib");
RUNTIME_COMPILER_LINKLIBRARY("G:\libs\gloox\libs\gloox-1.0d.lib");
#else
RUNTIME_COMPILER_LINKLIBRARY("Qt5Core.lib");
RUNTIME_COMPILER_LINKLIBRARY("Qt5Network.lib");
RUNTIME_COMPILER_LINKLIBRARY("Qt5Gui.lib");
RUNTIME_COMPILER_LINKLIBRARY("Qt5Widgets.lib");
RUNTIME_COMPILER_LINKLIBRARY("G:\libs\gloox\libs\gloox-1.0.lib");
#endif
#else

#endif

SETUP_PROJECT_DEF
#define GLOOX_IMPORTS
#include "gloox/loghandler.h"
#include "gloox/connectionlistener.h"
#include "gloox/messagesessionhandler.h"
#include "gloox/messageeventhandler.h"
#include "gloox/messageeventfilter.h"
#include "gloox/messagehandler.h"
#include "gloox/client.h"
#include "gloox/chatstatehandler.h"
#include "gloox/chatstatefilter.h"


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

	using namespace gloox;
	class XmppClient : public Node, MessageSessionHandler, ConnectionListener, LogHandler,
		MessageEventHandler, MessageHandler, ChatStateHandler
	{
		std::shared_ptr<gloox::Client> xmpp_client;
		MessageSession *m_session;
		MessageEventFilter* m_messageEventFilter;
		ChatStateFilter* m_chatStateFilter;
	public:
		virtual void onConnect();
		virtual void onDisconnect(ConnectionError e);
		virtual bool onTLSConnect(const CertInfo& info);
		virtual void handleMessage(const Message& msg, MessageSession * session);
		virtual void handleMessageEvent(const JID& from, MessageEventType messageEvent);
		virtual void handleChatState(const JID& from, ChatStateType state);
		virtual void handleMessageSession(MessageSession *session);
		virtual void handleLog(LogLevel level, LogArea area, const std::string& message);

		void on_msgReceived(std::string& msg);
		XmppClient();
		virtual void Init(bool firstInit);
		virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat& img, cv::cuda::Stream& stream /* = cv::cuda::Stream::Null() */);
	};

	
}
