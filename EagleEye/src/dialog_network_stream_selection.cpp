#include "dialog_network_stream_selection.h"
#include "ui_dialog_network_stream_selection.h"
#include <qlistwidget.h>
dialog_network_stream_selection::dialog_network_stream_selection(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::dialog_network_stream_selection)
{
    ui->setupUi(this);
    updateParameterPtr("url history", &url_history);
    variable_storage::instance().load_parameters(this);
    ui->list_url_history->installEventFilter(this);
    connect(ui->list_url_history, SIGNAL(itemClicked(QListWidgetItem*)), this, SLOT(on_item_clicked(QListWidgetItem*)));
    refresh_history();
}

dialog_network_stream_selection::~dialog_network_stream_selection()
{
    url_history.clear();
    for(int i = 0; i < ui->list_url_history->count(); ++i)
    {
        url_history.insert(ui->list_url_history->item(i)->text().toStdString());
    }
    variable_storage::instance().save_parameters(this);
    delete ui;
}

void dialog_network_stream_selection::accept()
{
    url = ui->txt_url_entry->toPlainText();
    url_history.insert(url.toStdString());
    ui->list_url_history->addItem(new QListWidgetItem(url));
    refresh_history();
    this->close();
}

void dialog_network_stream_selection::cancel()
{
    
}
void dialog_network_stream_selection::refresh_history()
{
    ui->list_url_history->clear();
    for(auto& itr : url_history)
    {
        ui->list_url_history->addItem(new QListWidgetItem(QString::fromStdString(itr)));
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