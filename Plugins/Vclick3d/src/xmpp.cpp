#include "Persistence/TextSerializer.hpp"
#include "xmpp.h"
#include "gloox/disco.h"
#include "gloox/message.h"
#include "gloox/gloox.h"
#include "gloox/siprofileft.h"
#include "gloox/siprofilefthandler.h"
#include "gloox/bytestreamdatahandler.h"
#include "gloox/chatstatefilter.h"
#include "gloox/socks5bytestreamserver.h"
#include "gloox/messageeventfilter.h"
#include <boost/algorithm/string/predicate.hpp>
using namespace gloox;
using namespace EagleLib;

void XmppClient::onConnect()
{
    LOG_TRACE;
}
void XmppClient::onDisconnect(ConnectionError e)
{
    LOG_TRACE;
}
bool XmppClient::onTLSConnect(const CertInfo& info)
{
    LOG_TRACE;
    return true;
}
void XmppClient::handleMessage(const Message& msg, MessageSession * session)
{
    LOG_TRACE;
    auto body = msg.body();
    NODE_LOG(debug) << "Received message " << body;
    updateParameter<std::string>("Message", body);
    auto nodes = getNodesInScope();

    if (boost::starts_with(body, "SetParameter\n"))
    {
        std::stringstream ss(body);
        std::string line;
        std::getline(ss, line);
        while (ss.good())
        {
            try
            {
                auto inputParam = Parameters::Persistence::Text::DeSerialize(&ss);
                for (auto node : nodes)
                {
                    if (node->fullTreeName == inputParam->GetTreeRoot())
                    {
                        for (auto param : node->parameters)
                        {
                            if (param->GetName() == inputParam->GetName())
                            {
                                param->Update(inputParam);
                            }

                        }
                    }

                }
            }
            catch (...)
            {

            }
        }
    }
}
void XmppClient::handleMessageEvent(const JID& from, MessageEventType messageEvent)
{
    LOG_TRACE;
}
void XmppClient::handleChatState(const JID& from, ChatStateType state)
{
    LOG_TRACE;
}
void XmppClient::handleMessageSession(MessageSession *session)
{
    LOG_TRACE;
    m_session.push_back(session);
    session->registerMessageHandler(this);
    m_messageEventFilter = new MessageEventFilter(session);
    m_messageEventFilter->registerMessageEventHandler(this);
    m_chatStateFilter = new ChatStateFilter(session);
    m_chatStateFilter->registerChatStateHandler(this);
    session->send("IP:68.100.56.64");
    std::stringstream ss;
    auto nodes = getNodesInScope();
    for (auto node : nodes)
    {
        for (auto param : node->parameters)
        {
            Parameters::Persistence::Text::Serialize(&ss, param.get());
            //ss << param->GetTreeName() << ":" << param->GetTypeInfo().name() << "\n";
        }
    }
    session->send(ss.str());
}
void XmppClient::handleLog(LogLevel level, LogArea area, const std::string& message)
{
    LOG_TRACE;
    switch (level)
    {
    case LogLevelDebug:
    {
                          NODE_LOG(debug) << message;
    }
    case LogLevelError:
    {
                          NODE_LOG(error) << message;
    }
    case LogLevelWarning:
    {
                            NODE_LOG(warning) << message;
    }

    }
}

void XmppClient::on_msgReceived(std::string& msg)
{

}

void XmppClient::Init(bool firstInit)
{
    if (firstInit)
    {
        updateParameter<std::string>("Jabber id", "dtmoodie");
        updateParameter<std::string>("Password", "12369pp");
        updateParameter<std::string>("Jabber server", "jabber.iitsp.com");
        updateParameter<unsigned short>("Server port", 5222);
        addInputParameter<cv::cuda::GpuMat>("Input point cloud");
        RegisterParameterCallback("Input point cloud", boost::bind(&XmppClient::_sendPointCloud, this));
    }

}
void XmppClient::_sendPointCloud()
{
    //Parameters::UI::ProcessingThreadCallbackService::Instance()->post(boost::bind(&XmppClient::sendPointCloud, this));
}
void XmppClient::sendPointCloud()
{
    auto gpuMat = getParameter<cv::cuda::GpuMat>("Input point cloud")->Data();

}
cv::cuda::GpuMat XmppClient::doProcess(cv::cuda::GpuMat& img, cv::cuda::Stream& stream)
{
    if (parameters[0]->changed || parameters[1]->changed || parameters[2]->changed || parameters[3]->changed)
    {
        gloox::JID jid(*getParameter<std::string>(0)->Data() + "@" + *getParameter<std::string>(2)->Data());;
        xmpp_client.reset(new gloox::Client(jid, *getParameter<std::string>(1)->Data(), *getParameter<unsigned short>(3)->Data()));
        xmpp_client->registerConnectionListener(this);
        xmpp_client->registerMessageSessionHandler(this, 0);
        xmpp_client->disco()->setVersion("messageTest", GLOOX_VERSION, "Linux");
        xmpp_client->disco()->setIdentity("client", "bot");
        xmpp_client->disco()->addFeature(XMLNS_CHAT_STATES);
        xmpp_client->logInstance().registerLogHandler(LogLevelDebug, LogAreaAll, this);
        if (!xmpp_client->connect(false))
        {
            NODE_LOG(error) << "Unable to connect";
        }
        parameters[0]->changed = false;
        parameters[1]->changed = false;
        parameters[2]->changed = false;
        parameters[3]->changed = false;
    }
    if (xmpp_client)
    {
        xmpp_client->recv(0);
    }
    return img;
}


NODE_DEFAULT_CONSTRUCTOR_IMPL(XmppClient, Utility)
