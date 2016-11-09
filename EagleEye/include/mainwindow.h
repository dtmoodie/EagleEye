#ifndef MAINWINDOW_H
#define MAINWINDOW_H
#include <QMainWindow>
#include "EagleLib/Signals.h"
#include <EagleLib/Nodes/Node.h>


#include <qtimer.h>
#include "NodeListDialog.h"
#include <qgraphicsscene.h>
#include <qgraphicsview.h>
#include "NodeView.h"
#include <qlist.h>
#include <vector>
#include <boost/thread.hpp>
#include "rccsettingsdialog.h"
#include "user_interface_persistence.h"
#include "plotwizarddialog.h"
#include <QtGui/qopenglcontext.h>
#include <MetaObject/Signals/RelayManager.hpp>


#include <qtimer.h>

#include <functional>

namespace EagleLib
{
    class IDataStream;
}

namespace Ui {
class MainWindow;
}
class SettingDialog;
class bookmark_dialog;

class MainWindow : public QMainWindow, public mo::IMetaObject, public UIPersistence
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();

    MO_BEGIN(MainWindow)
        MO_SIGNAL(void, StartThreads);
        MO_SIGNAL(void, StopThreads);
        MO_SIGNAL(void, PauseThreads);
        MO_SIGNAL(void, ResumeThreads);
        MO_SLOT(void, parameter_added, EagleLib::Nodes::Node*);
        MO_SLOT(void, node_update, EagleLib::Nodes::Node*);
    MO_END;

    
    void onCompileLog(const std::string& msg, int level);
    virtual void closeEvent(QCloseEvent *event);
    void processingThread_uiCallback(boost::function<void(void)> f, std::pair<void*, mo::TypeInfo> source);
    void process_log_message(boost::log::trivial::severity_level severity, std::string message);
    std::vector<mo::IParameter*> GetParameters()
    {
        return this->mo::IMetaObject::GetParameters();
    }
public slots:
    void load_file(QString file, QString preferred_loader = "");
private slots:
    void on_pushButton_clicked();
    void onTimeout();
    
    void onSelectionChanged(QGraphicsProxyWidget* widget);
    void log(QString message);
    
    void stopProcessingThread();
    void startProcessingThread();
    void onWidgetDeleted(QNodeWidget* widget);
    void onWidgetDeleted(DataStreamWidget* widget);
    void onSaveClicked();
    void onLoadClicked();
    void onLoadFileClicked();
    void onLoadDirectoryClicked();
    void onLoadPluginClicked();
    void addNode(EagleLib::Nodes::Node::Ptr node);
    void updateLines();
    void uiNotifier();
    void onUiUpdate();
    //void on_NewParameter(EagleLib::Nodes::Node* node);
    //void newParameter(EagleLib::Nodes::Node* node);
    void displayRCCSettings();
    void onPlotAdd(PlotWindow* plot);
    void onPlotRemove(PlotWindow* plot);
    void on_actionLog_settings_triggered();
    void on_actionOpen_Network_triggered();
    void on_actionBookmarks_triggered();

    void on_btnClear_clicked();
    void on_uiCallback(boost::function<void()> f, std::pair<void*, mo::TypeInfo> source);
    

    void on_btnStart_clicked();

    void on_btnStop_clicked();
    void on_nodeUpdate(EagleLib::Nodes::Node* node);
    
    void on_persistence_timeout();

signals:
    //void onNewParameter(EagleLib::Nodes::Node* node);
    void eLog(QString message);
    void oglDisplayImage(std::string name, cv::cuda::GpuMat img);
    void qtDisplayImage(std::string name, cv::Mat img);
    void qtDisplayImage(boost::function<cv::Mat(void)> function, EagleLib::Nodes::Node* node);
    void uiCallback(boost::function<void()> f, std::pair<void*, mo::TypeInfo> source);
    void uiNeedsUpdate();
    void pluginLoaded();


private:
    void onNodeAdd(std::string name);
    void onError(const std::string& error);
    void onStatus(const std::string& status);
    void processThread();
    bool processingThreadActive;

    bookmark_dialog*                                    bookmarks;
    bool                                                dirty;
    Ui::MainWindow *                                    ui;
    QTimer*                                             fileMonitorTimer;
    NodeListDialog*                                     nodeListDialog;
    QGraphicsScene*                                     nodeGraph;
    NodeView*                                           nodeGraphView;
    QGraphicsProxyWidget*                               currentSelectedNodeWidget;
    QGraphicsProxyWidget*                               currentSelectedStreamWidget;
    rcc::weak_ptr<EagleLib::Nodes::Node>                currentNode;
    rcc::weak_ptr<EagleLib::IDataStream>                current_stream;
    
    std::vector<EagleLib::Nodes::Node::Ptr>             parentList;
    boost::timed_mutex                                  parentMtx;
    std::vector<QNodeWidget*>                           widgets;
    std::vector<DataStreamWidget*>                      data_stream_widgets;
    RCCSettingsDialog*                                  rccSettings;
    std::map<std::string, cv::Vec2f>                    positionMap;
    PlotWizardDialog*                                   plotWizardDialog;
    SettingDialog*                                      settingsDialog;
    QOpenGLContext*                                     processing_thread_context;
    QWindow*                                            processing_thread_upload_window;
    std::shared_ptr<mo::Connection>                     new_parameter_connection;
    std::shared_ptr<mo::Connection>                     dirty_flag_connection;
    std::vector<rcc::shared_ptr<EagleLib::IDataStream>> data_streams;
    std::shared_ptr<mo::Connection>                     logging_connection;
    std::string                                         file_load_path;
    std::string                                         dir_load_path;
    QTimer*                                             persistence_timer;

    // All signals from the user get directed through this manager so that 
    // they can all be attached to a serialization sink for recording user interaction
    mo::RelayManager                                    _ui_manager;
    std::vector<std::shared_ptr<mo::Connection>>        _signal_connections;
};

#endif // MAINWINDOW_H
