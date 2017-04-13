#include "bookmark_dialog.h"
#include "ui_bookmark_dialog.h"
#include "MetaObject/Parameters/detail/TypedParameterPtrImpl.hpp"
bookmark_dialog::bookmark_dialog(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::bookmark_dialog)
{
    ui->setupUi(this);
    bookmarks_param.UpdatePtr(&bookmarks);
    history_param.UpdatePtr(&history);
    VariableStorage::Instance()->LoadParams(this, "bookmarks");
    update();
    QObject::connect(ui->list_bookmarks, SIGNAL(itemDoubleClicked(QListWidgetItem*)), this, SLOT(on_file_selected(QListWidgetItem*)));
    QObject::connect(ui->list_history, SIGNAL(itemDoubleClicked(QListWidgetItem*)), this,SLOT(on_file_selected(QListWidgetItem*)));
    
}

bookmark_dialog::~bookmark_dialog()
{
    VariableStorage::Instance()->SaveParams(this, "bookmarks");
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
std::vector<mo::IParameter*> bookmark_dialog::GetParameters()
{
    std::vector<mo::IParameter*> output;
    output.push_back(&bookmarks_param);
    output.push_back(&history_param);
    return output;
}
