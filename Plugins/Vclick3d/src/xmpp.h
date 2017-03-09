#include "EagleLib/Nodes/Node.h"
#include "ObjectInterfacePerModule.h"
#include "EagleLib/ObjectDetection.hpp"

#ifdef HAVE_GLOOX
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
RUNTIME_COMPILER_LINKLIBRARY("G:/libs/gloox/libs/gloox-1.0d.lib");
#else
RUNTIME_COMPILER_LINKLIBRARY("G:/libs/gloox/libs/gloox-1.0.lib");
#endif
#else

#endif

using namespace gloox;
namespace EagleLib
{
    namespace Nodes
    {
    
    
    class XmppClient : public Node, MessageSessionHandler, ConnectionListener, LogHandler,
        MessageEventHandler, MessageHandler, ChatStateHandler
    {
        std::shared_ptr<gloox::Client> xmpp_client;
        std::list<MessageSession*> m_session;
        MessageEventFilter* m_messageEventFilter;
        ChatStateFilter* m_chatStateFilter;
    public:
        MO_DERIVE(XmppClient, Node)
            INPUT(std::vector<DetectedObject2d>, detections, nullptr)
            PARAM(std::string, jid, "dtmoodie")
            PARAM(std::string, pass, "12369pp")
            PARAM(std::string, server, "jabber.iitsp.com")
            PARAM(uint16_t, port, 5222)
        MO_END;

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
    protected:
        bool ProcessImpl();

        //virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat& img, cv::cuda::Stream& stream /* = cv::cuda::Stream::Null() */);
    };
    }
}
#endif
