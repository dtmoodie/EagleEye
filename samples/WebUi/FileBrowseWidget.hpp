#pragma once
#include "Wt/WContainerWidget"
#include "Wt/WSignal"

#include <boost/filesystem.hpp>
namespace Wt
{
    class WPushButton;
    class WLineEdit;
    class WSuggestionPopup;
    class WFileUpload;

    class AutoCompleteFileWidget: public WContainerWidget
    {
    public:
        AutoCompleteFileWidget(WContainerWidget* parent = 0,
                         const std::string& upload_dir = "/tmp/",
                         const boost::filesystem::path & current_dir = boost::filesystem::current_path());


        /*!
         * \brief fileSelected
         * \return a reference to the event signal object that is emitted when a selection is made
         */
        EventSignal<>& fileSelected();
        std::string hostFile();
    protected:
        void onFilterModel(const WString& data_);
        WLineEdit* _txt_manual_entry;
        WSuggestionPopup* _sp;
        boost::filesystem::path _current_path;
        EventSignal<> _sig_file_selected;
        //Signal<std::string, std::string> _sig_file_selected;
    };


    // Creates a widget that allows text entry with auto complete for files on the
    // serving computer, as well as a button that allows for uploading files from
    // the user's computer
    // ----------------------------------------------
    //  [ --- text entry area --- ] [ Browse ]
    // ----------------------------------------------
    class FileBrowseWidget: public AutoCompleteFileWidget
    {
    public:
        /*!
         * \brief FileBrowseWidget
         * \param parent
         * \param upload_dir is the directory where user uploaded files will be stored
         * \param current_dir
         */
        FileBrowseWidget(WContainerWidget* parent = 0,
                         const std::string& upload_dir = "/tmp/",
                         const boost::filesystem::path & current_dir = boost::filesystem::current_path());
        std::string clientFile();
    private:

        WFileUpload* _btn_browse;
    };
}
