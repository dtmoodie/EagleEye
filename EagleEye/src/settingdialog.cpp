#include "settingdialog.h"
#include "ui_settingdialog.h"
#include <boost/log/core.hpp>
#include <boost/log/trivial.hpp>
#include <boost/log/expressions.hpp>
#include <boost/log/sinks.hpp>
#include <boost/log/attributes.hpp>
#include <boost/log/common.hpp>
#include <boost/log/exceptions.hpp>
#include <boost/log/utility/setup/file.hpp>

SettingDialog::SettingDialog(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::SettingDialog)
{
    ui->setupUi(this);
    connect(ui->debugLevel, SIGNAL(currentIndexChanged(int)), this, SLOT(on_debugLevel_indexChanged(int)));
}

SettingDialog::~SettingDialog()
{
    delete ui;
}

void SettingDialog::on_debugLevel_indexChanged(int value)
{
	
    switch(value)
    {
    case 0:
        boost::log::core::get()->set_filter(boost::log::trivial::severity >= boost::log::trivial::trace);
        break;
    case 1:
        boost::log::core::get()->set_filter(boost::log::trivial::severity >= boost::log::trivial::debug);
        break;
    case 2:
        boost::log::core::get()->set_filter(boost::log::trivial::severity >= boost::log::trivial::info);
        break;
    case 3:
        boost::log::core::get()->set_filter(boost::log::trivial::severity >= boost::log::trivial::warning);
        break;
    case 4:
        boost::log::core::get()->set_filter(boost::log::trivial::severity >= boost::log::trivial::error);
        break;
    case 5:
        boost::log::core::get()->set_filter(boost::log::trivial::severity >= boost::log::trivial::fatal);
        break;
    }
}
