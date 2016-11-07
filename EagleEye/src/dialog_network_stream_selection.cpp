#include "dialog_network_stream_selection.h"
#include "ui_dialog_network_stream_selection.h"
#include <qlistwidget.h>

#include <EagleLib/Nodes/IFrameGrabber.hpp>

dialog_network_stream_selection::dialog_network_stream_selection(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::dialog_network_stream_selection)
{
    ui->setupUi(this);
    updateParameterPtr("url history", &url_history);
    variable_storage::instance().load_parameters(this);
    ui->list_url_history->installEventFilter(this);
    QObject::connect(ui->list_url_history, SIGNAL(itemClicked(QListWidgetItem*)), this, SLOT(on_item_clicked(QListWidgetItem*)));

    auto constructors = EagleLib::ObjectManager::Instance().GetConstructorsForInterface(IID_FrameGrabber);
    for(auto constructor : constructors)
    {
        auto info = constructor->GetObjectInfo();
        if(info)
        {
            auto fg_info = dynamic_cast<EagleLib::FrameGrabberInfo*>(info);
            if(fg_info)
            {
                auto devices = fg_info->ListLoadableDocuments();
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
    variable_storage::instance().save_parameters(this);
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
