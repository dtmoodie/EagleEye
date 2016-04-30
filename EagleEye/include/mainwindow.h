#ifndef MAINWINDOW_H
#define MAINWINDOW_H
#include <QMainWindow>
#include "EagleLib/Signals.h"
#include <EagleLib/nodes/Node.h>


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
#include <signals/connection.h>
#include <signals/signaler.h>
#include <signals/signal_manager.h>

#include <qtimer.h>

#include <functional>

namespace EagleLib
{
    class DataStream;
}

namespace Ui {
class MainWindow;
}
class SettingDialog;
class bookmark_dialog;

class MainWindow : public QMainWindow, public user_interface_persistence//, Signals::signaler
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();
    void oglDisplay(cv::cuda::GpuMat img, EagleLib::Nodes::Node *node);
    void qtDisplay(cv::Mat img, EagleLib::Nodes::Node *node);
    void onCompileLog(const std::string& msg, int level);
    virtual void closeEvent(QCloseEvent *event);
    void processingThread_uiCallback(boost::function<void(void)> f, std::pair<void*, Loki::TypeInfo> source);
	void process_log_message(boost::log::trivial::severity_level severity, std::string message);
private slots:
    void on_pushButton_clicked();
    void onTimeout();
    
	void onSelectionChanged(QGraphicsProxyWidget* widget);
    void log(QString message);
    void onOGLDisplay(std::string name, cv::cuda::GpuMat img);
    void onQtDisplay(std::string name, cv::Mat img);
    void onQtDisplay(boost::function<cv::Mat(void)> function, EagleLib::Nodes::Node* node);
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
    void on_NewParameter(EagleLib::Nodes::Node* node);
    void newParameter(EagleLib::Nodes::Node* node);
    void displayRCCSettings();
    void onPlotAdd(PlotWindow* plot);
    void onPlotRemove(PlotWindow* plot);
    void on_actionLog_settings_triggered();
    void on_actionOpen_Network_triggered();
    void on_actionBookmarks_triggered();

    void on_btnClear_clicked();
    void on_uiCallback(boost::function<void()> f, std::pair<void*, Loki::TypeInfo> source);
    

    void on_btnStart_clicked();

    void on_btnStop_clicked();
    void on_nodeUpdate(EagleLib::Nodes::Node* node);
    void load_file(QString file);
    void on_persistence_timeout();

signals:
    void onNewParameter(EagleLib::Nodes::Node* node);
    void eLog(QString message);
    void oglDisplayImage(std::string name, cv::cuda::GpuMat img);
    void qtDisplayImage(std::string name, cv::Mat img);
    void qtDisplayImage(boost::function<cv::Mat(void)> function, EagleLib::Nodes::Node* node);
    void uiCallback(boost::function<void()> f, std::pair<void*, Loki::TypeInfo> source);
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
	NodeView*	                                        nodeGraphView;
	QGraphicsProxyWidget*                               currentSelectedNodeWidget;
    QGraphicsProxyWidget*                               currentSelectedStreamWidget;
    EagleLib::Nodes::Node::Ptr                          currentNode;
    EagleLib::DataStream::Ptr                           current_stream;
    
    std::vector<EagleLib::Nodes::Node::Ptr>             parentList;
    boost::timed_mutex                                  parentMtx;
    std::vector<QNodeWidget*>                           widgets;
    std::vector<DataStreamWidget*>                       data_stream_widgets;
    //boost::thread                                       processingThread;
    RCCSettingsDialog*                                  rccSettings;
    std::map<std::string, cv::Vec2f>                    positionMap;
    PlotWizardDialog*                                   plotWizardDialog;
	SettingDialog*                                      settingsDialog;
    QOpenGLContext*                                     processing_thread_context;
    QWindow*                                            processing_thread_upload_window;
    std::shared_ptr<Signals::connection>                new_parameter_connection;
    std::shared_ptr<Signals::connection>                dirty_flag_connection;
    std::vector<std::shared_ptr<EagleLib::DataStream>>  data_streams;
    std::shared_ptr<Signals::connection>                logging_connection;
    std::string file_load_path;
    std::string dir_load_path;

    QTimer* persistence_timer;
    SIG_DEF(StartThreads);
    SIG_DEF(StopThreads);
	// All signals from the user get directed through this manager so that 
	// they can all be attached to a serialization sink for recording user interaction
	Signals::signal_manager								_ui_manager;
	std::vector<std::shared_ptr<Signals::connection>>   _signal_connections;
};

#endif // MAINWINDOW_H
