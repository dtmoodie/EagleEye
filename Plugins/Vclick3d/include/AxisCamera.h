#pragma once
#include "EagleLib/nodes/Node.h"
#include "qnetworkaccessmanager.h"
#include "qnetworkreply.h"
#include "qtcpsocket.h"
#include "ObjectInterfacePerModule.h"
#include <EagleLib/Project_defs.hpp>


#ifdef _MSC_VER
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
#else

#endif

SETUP_PROJECT_DEF



namespace EagleLib
{
    namespace Nodes
    {
    class AxisSocket: public QObject
    {
        Q_OBJECT
        QNetworkAccessManager* network_manager;
        QNetworkReply* network_request;        
    public:
        Parameters::ITypedParameter<int>* zoom;
        Parameters::ITypedParameter<int>* focus;

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
        std::shared_ptr<Signals::connection> zoomConnection;
        std::shared_ptr<Signals::connection> panConnection;
        std::shared_ptr<Signals::connection> tiltConnection;
        std::shared_ptr<Signals::connection> ipConnection;
        std::shared_ptr<Signals::connection> usernameConnection;
        std::shared_ptr<Signals::connection> passwordConnection;
        boost::shared_ptr<AxisSocket> socket;
        
        void on_zoomRequest();
        void on_panRequest();
        void on_tiltRequest();
        void on_addressChange();
        void on_credentialChange();
        void get_position();
        Parameters::ITypedParameter<int>* currentZoom;
        Parameters::ITypedParameter<int>* currentFocus;

    public:
        AxisCamera();
        ~AxisCamera();
        virtual void NodeInit(bool firstInit);
        virtual void Serialize(ISimpleSerializer* pSerializer);
        virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat& img, cv::cuda::Stream& stream = cv::cuda::Stream::Null());
        virtual bool SkipEmpty() const;

    };
    }
}
