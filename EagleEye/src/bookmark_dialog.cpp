#include "bookmark_dialog.h"
#include "ui_bookmark_dialog.h"

bookmark_dialog::bookmark_dialog(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::bookmark_dialog)
{
    ui->setupUi(this);
    updateParameterPtr("bookmarks", &bookmarks);
    updateParameterPtr("history", &history);

    variable_storage::instance().load_parameters(this);
    update();
    QObject::connect(ui->list_bookmarks, SIGNAL(itemDoubleClicked(QListWidgetItem*)), this, SLOT(on_file_selected(QListWidgetItem*)));
    QObject::connect(ui->list_history, SIGNAL(itemDoubleClicked(QListWidgetItem*)), this,SLOT(on_file_selected(QListWidgetItem*)));
    
}

bookmark_dialog::~bookmark_dialog()
{
    variable_storage::instance().save_parameters(this);
    delete ui;
}

void bookmark_dialog::update()
{
    ui->list_history->clear();
    ui->list_bookmarks->clear();
    for(auto& itr: history)
        ui->list_history->addItem(new QListWidgetItem(QString::fromStdString(itr)));
    for(auto& itr: bookmarks)
        ui->list_bookmarks->addItem(new QListWidgetItem(QString::fromStdString(itr)));
}

void bookmark_dialog::append_history(std::string dir)
{
    history.insert(dir);
    update();
}
void bookmark_dialog::on_file_selected(QListWidgetItem* item)
{
    auto name = item->text();
    if(sender() == ui->list_bookmarks)
    {
        
    }
    if(sender() == ui->list_history)
    {
        
    }
    emit open_file(name);
}