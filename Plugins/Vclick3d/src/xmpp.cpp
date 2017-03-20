#ifdef HAVE_GLOOX
#include <Aquila/Nodes/NodeInfo.hpp>
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
using namespace aq;
using namespace aq::Nodes;

void XmppClient::onConnect()
{
    
}
void XmppClient::onDisconnect(ConnectionError e)
{
    
}
bool XmppClient::onTLSConnect(const CertInfo& info)
{
    
    return true;
}
void XmppClient::handleMessage(const Message& msg, MessageSession * session)
{
    //auto body = msg.body();
    //LOG(trace) << "Received message " << body;
    //updateParameter<std::string>("Message", body);
    //auto nodes = getNodesInScope();

    /*if (boost::starts_with(body, "SetParameter\n"))
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
                    if (node->getFullTreeName() == inputParam->GetTreeRoot())
                    {
                        auto parameters = node->getParameters();
                        for (auto param : parameters)
                        {
                            if (param->GetName() == inputParam->GetName())
                            {
                                param->Update(inputParam.get());
                            }

                        }
                    }

                }
            }
            catch (...)
            {

            }
        }
    }*/
}
void XmppClient::handleMessageEvent(const JID& from, MessageEventType messageEvent)
{
    
}
void XmppClient::handleChatState(const JID& from, ChatStateType state)
{
    
}
void XmppClient::handleMessageSession(MessageSession *session)
{
    
    m_session.push_back(session);
    session->registerMessageHandler(this);
    m_messageEventFilter = new MessageEventFilter(session);
    m_messageEventFilter->registerMessageEventHandler(this);
    m_chatStateFilter = new ChatStateFilter(session);
    m_chatStateFilter->registerChatStateHandler(this);
    session->send("IP:68.100.56.64");
    std::stringstream ss;
    /*auto nodes = getNodesInScope();
    for (auto node : nodes)
    {
        auto parameters = node->getParameters();
        for (auto param : parameters)
        {
            Parameters::Persistence::Text::Serialize(&ss, param);
            //ss << param->GetTreeName() << ":" << param->GetTypeInfo().name() << "\n";
        }
    }*/
    session->send(ss.str());
}
void XmppClient::handleLog(LogLevel level, LogArea area, const std::string& message)
{
    
    switch (level)
    {
    case LogLevelDebug:
    {
         LOG(debug) << message;
    }
    case LogLevelError:
    {
        LOG(error) << message;
    }
    case LogLevelWarning:
    {
        LOG(warning) << message;
    }

    }
}

void XmppClient::on_msgReceived(std::string& msg)
{

}


void XmppClient::_sendPointCloud()
{
    //Parameters::UI::ProcessingThreadCallbackService::Instance()->post(boost::bind(&XmppClient::sendPointCloud, this));
}
void XmppClient::sendPointCloud()
{


}

/*cv::cuda::GpuMat XmppClient::doProcess(cv::cuda::GpuMat& img, cv::cuda::Stream& stream)
{
    if (_parameters[0]->changed || _parameters[1]->changed || _parameters[2]->changed || _parameters[3]->changed)
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
        _parameters[0]->changed = false;
        _parameters[1]->changed = false;
        _parameters[2]->changed = false;
        _parameters[3]->changed = false;
    }
    if (xmpp_client)
    {
        xmpp_client->recv(0);
    }
    return img;
}*/
bool XmppClient::ProcessImpl()
{
    if(jid_param._modified || pass_param._modified || server_param._modified || port_param._modified)
    {
        gloox::JID jid(jid + "@" + server);
        xmpp_client.reset(new gloox::Client(jid, pass, port));
        xmpp_client->registerConnectionListener(this);
        xmpp_client->registerMessageSessionHandler(this, 0);
        xmpp_client->disco()->setVersion("messageTest", GLOOX_VERSION, "Linux");
        xmpp_client->disco()->setIdentity("client", "bot");
        xmpp_client->disco()->addFeature(XMLNS_CHAT_STATES);
        xmpp_client->logInstance().registerLogHandler(LogLevelDebug, LogAreaAll, this);
        if (!xmpp_client->connect(false))
        {
            LOG(error) << "Unable to connect";
        }
    }
    return true;
}

MO_REGISTER_CLASS(XmppClient)
#endif
