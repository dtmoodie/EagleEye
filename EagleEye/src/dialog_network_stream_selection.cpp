#include "dialog_network_stream_selection.h"
#include "ui_dialog_network_stream_selection.h"
#include <qlistwidget.h>
#include "MetaObject/Parameters/detail/TypedParameterPtrImpl.hpp"
#include <Aquila/framegrabbers/IFrameGrabber.hpp>

dialog_network_stream_selection::dialog_network_stream_selection(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::dialog_network_stream_selection)
{
    ui->setupUi(this);
    url_history_param.UpdatePtr(&url_history);
    VariableStorage::Instance()->LoadParams(this, "Network Streams");
    ui->list_url_history->installEventFilter(this);
    QObject::connect(ui->list_url_history, SIGNAL(itemClicked(QListWidgetItem*)), this, SLOT(on_item_clicked(QListWidgetItem*)));

    auto constructors = mo::MetaObjectFactory::instance()->getConstructors(aq::Nodes::IFrameGrabber::s_interfaceID);
    for(auto constructor : constructors)
    {
        auto info = constructor->GetObjectInfo();
        if(info)
        {
            auto fg_info = dynamic_cast<aq::Nodes::FrameGrabberInfo*>(info);
            if(fg_info)
            {
                auto devices = fg_info->ListLoadablePaths();
                if(devices.size())
                {
                    for(auto& device : devices)
                    {
                        url_history.insert(std::make_pair(device, std::string()));
                    }
                }
            }
        }
    }
    refresh_history();
}

dialog_network_stream_selection::~dialog_network_stream_selection()
{
    url_history.clear();
    for(int i = 0; i < ui->list_url_history->count(); ++i)
    {
        url_history.insert(std::make_pair(ui->list_url_history->item(i)->text().toStdString(),std::string("")));
    }
    VariableStorage::Instance()->SaveParams(this, "Network Stream");
    delete ui;
}

void dialog_network_stream_selection::accept()
{
    url = ui->txt_url_entry->toPlainText();
    preferred_loader = ui->txt_frame_grabber_overload->toPlainText();
    url_history.insert(std::make_pair(url.toStdString(), preferred_loader.toStdString()));
    ui->list_url_history->addItem(new QListWidgetItem(url));
    refresh_history();
    this->accepted = true;
    this->close();
}

void dialog_network_stream_selection::cancel()
{
    this->accepted = false;
    this->close();
}

void dialog_network_stream_selection::refresh_history()
{
    ui->list_url_history->clear();
    for(auto& itr : url_history)
    {
        ui->list_url_history->addItem(new QListWidgetItem(QString::fromStdString(itr.first)));
    }
}
void dialog_network_stream_selection::on_item_clicked(QListWidgetItem* item)
{
    ui->txt_url_entry->setPlainText(item->text());
}
bool dialog_network_stream_selection::eventFilter(QObject *object, QEvent *event)
{
    if(object == ui->list_url_history)
    {
        if(event->type() == QEvent::KeyPress)
        {
            QKeyEvent* key_event = static_cast<QKeyEvent*>(event);
            if(key_event->key() == Qt::Key_Delete)
            {
                auto items = ui->list_url_history->selectedItems();
                for(auto item: items)
                {
                    delete item;
                }
                return true;
            }
        }
    }
    return false;
}
std::vector<mo::IParam*> dialog_network_stream_selection::GetParameters()
{
    std::vector<mo::IParam*> output;
    output.push_back(&url_history_param);
    return output;
}
