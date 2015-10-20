#include "nodes/Node.h"
#include "ObjectInterfacePerModule.h"
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


#ifdef _MSC_VER
#ifdef _DEBUG
RUNTIME_COMPILER_LINKLIBRARY("G:\libs\gloox\libs\gloox-1.0d.lib");
#else
RUNTIME_COMPILER_LINKLIBRARY("G:\libs\gloox\libs\gloox-1.0.lib");
#endif
#else

#endif

using namespace gloox;
namespace EagleLib
{
	class XmppClient : public Node, MessageSessionHandler, ConnectionListener, LogHandler,
		MessageEventHandler, MessageHandler, ChatStateHandler
	{
		std::shared_ptr<gloox::Client> xmpp_client;
		std::list<MessageSession*> m_session;
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
		virtual void sendPointCloud();
		virtual void _sendPointCloud();
		void on_msgReceived(std::string& msg);
		XmppClient();
		virtual void Init(bool firstInit);
		virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat& img, cv::cuda::Stream& stream /* = cv::cuda::Stream::Null() */);
	};
}