#pragma once
#include <cereal/archives/json.hpp>
#include <cereal/types/vector.hpp>
#include <EagleLib/Nodes/Node.h>
#include <EagleLib/IDataStream.hpp>
#include <MetaObject/Parameters/IO/SerializationFunctionRegistry.hpp>
#include <MetaObject/Parameters/Buffers/IBuffer.hpp>
#include <MetaObject/Parameters/Buffers/BufferFactory.hpp>

#include <boost/lexical_cast.hpp>
namespace EagleLib
{
    struct InputInfo
    {
        std::string name;
        std::string type;
        bool sync = false;
        int buffer_size = -1;
        template<class AR> void serialize(AR& ar)
        {
            ar(CEREAL_NVP(name));
            ar(CEREAL_OPTIONAL_NVP(sync, false));
            ar(CEREAL_OPTIONAL_NVP(buffer_size, -1));
            ar(CEREAL_NVP(type));
        }
    };
    class EAGLE_EXPORTS JSONOutputArchive : public cereal::JSONOutputArchive
    {
    public:
        JSONOutputArchive(std::ostream & stream, Options const & options = Options::Default()) :
            cereal::JSONOutputArchive(stream, options)
        {
            
        }

        //! Destructor, flushes the JSON
        ~JSONOutputArchive() CEREAL_NOEXCEPT
        {
            
        }

        //! Saves some binary data, encoded as a base64 string, with an optional name
        /*! This will create a new node, optionally named, and insert a value that consists of
        the data encoded as a base64 string */
        void saveBinaryValue(const void * data, size_t size, const char * name = nullptr)
        {
            setNextName(name);
            writeName();

            auto base64string = cereal::base64::encode(reinterpret_cast<const unsigned char *>(data), size);
            saveValue(base64string);
        }

        void startNode()
        {
            writeName();
            itsNodeStack.push(NodeType::StartObject);
            itsNameCounter.push(0);
        }

        //! Designates the most recently added node as finished
        void finishNode()
        {
            switch (itsNodeStack.top())
            {
            case NodeType::StartArray:
                itsWriter.StartArray();
            case NodeType::InArray:
                itsWriter.EndArray();
                break;
            case NodeType::StartObject:
                itsWriter.StartObject();
            case NodeType::InObject:
                itsWriter.EndObject();
                break;
            }

            itsNodeStack.pop();
            itsNameCounter.pop();
        }

        //! Sets the name for the next node created with startNode
        void setNextName(const char * name)
        {
            itsNextName = name;
        }

        //! Saves a bool to the current node
        void saveValue(bool b) { itsWriter.Bool(b); }
        //! Saves an int to the current node
        void saveValue(int i) { itsWriter.Int(i); }
        //! Saves a uint to the current node
        void saveValue(unsigned u) { itsWriter.Uint(u); }
        //! Saves an int64 to the current node
        void saveValue(int64_t i64) { itsWriter.Int64(i64); }
        //! Saves a uint64 to the current node
        void saveValue(uint64_t u64) { itsWriter.Uint64(u64); }
        //! Saves a double to the current node
        void saveValue(double d) { itsWriter.Double(d); }
        //! Saves a string to the current node
        void saveValue(std::string const & s) { itsWriter.String(s.c_str(), static_cast<rapidjson::SizeType>(s.size())); }
        //! Saves a const char * to the current node
        void saveValue(char const * s) { itsWriter.String(s); }
        //! Saves a nullptr to the current node
        void saveValue(std::nullptr_t) { itsWriter.Null(); }

    protected:
        // Some compilers/OS have difficulty disambiguating the above for various flavors of longs, so we provide
        // special overloads to handle these cases.

        //! 32 bit signed long saving to current node
        template <class T, cereal::traits::EnableIf<sizeof(T) == sizeof(std::int32_t),
            std::is_signed<T>::value> = cereal::traits::sfinae> inline
            void saveLong(T l) { saveValue(static_cast<std::int32_t>(l)); }

        //! non 32 bit signed long saving to current node
        template <class T, cereal::traits::EnableIf<sizeof(T) != sizeof(std::int32_t),
            std::is_signed<T>::value> = cereal::traits::sfinae> inline
            void saveLong(T l) { saveValue(static_cast<std::int64_t>(l)); }

        //! 32 bit unsigned long saving to current node
        template <class T, cereal::traits::EnableIf<sizeof(T) == sizeof(std::int32_t),
            std::is_unsigned<T>::value> = cereal::traits::sfinae> inline
            void saveLong(T lu) { saveValue(static_cast<std::uint32_t>(lu)); }

        //! non 32 bit unsigned long saving to current node
        template <class T, cereal::traits::EnableIf<sizeof(T) != sizeof(std::int32_t),
            std::is_unsigned<T>::value> = cereal::traits::sfinae> inline
            void saveLong(T lu) { saveValue(static_cast<std::uint64_t>(lu)); }

    public:
#ifdef _MSC_VER
        //! MSVC only long overload to current node
        void saveValue(unsigned long lu) { saveLong(lu); };
#else // _MSC_VER
        //! Serialize a long if it would not be caught otherwise
        template <class T, cereal::traits::EnableIf<std::is_same<T, long>::value,
            !std::is_same<T, std::int32_t>::value,
            !std::is_same<T, std::int64_t>::value> = cereal::traits::sfinae> inline
            void saveValue(T t) { saveLong(t); }

        //! Serialize an unsigned long if it would not be caught otherwise
        template <class T, cereal::traits::EnableIf<std::is_same<T, unsigned long>::value,
            !std::is_same<T, std::uint32_t>::value,
            !std::is_same<T, std::uint64_t>::value> = cereal::traits::sfinae> inline
            void saveValue(T t) { saveLong(t); }
#endif // _MSC_VER

        //! Save exotic arithmetic as strings to current node
        /*! Handles long long (if distinct from other types), unsigned long (if distinct), and long double */
        template <class T, cereal::traits::EnableIf<std::is_arithmetic<T>::value,
            !std::is_same<T, long>::value,
            !std::is_same<T, unsigned long>::value,
            !std::is_same<T, std::int64_t>::value,
            !std::is_same<T, std::uint64_t>::value,
            (sizeof(T) >= sizeof(long double) || sizeof(T) >= sizeof(long long))> = cereal::traits::sfinae> inline
            void saveValue(T const & t)
        {
            std::stringstream ss; ss.precision(std::numeric_limits<long double>::max_digits10);
            ss << t;
            saveValue(ss.str());
        }

        //! Write the name of the upcoming node and prepare object/array state
        /*! Since writeName is called for every value that is output, regardless of
        whether it has a name or not, it is the place where we will do a deferred
        check of our node state and decide whether we are in an array or an object.

        The general workflow of saving to the JSON archive is:

        1. (optional) Set the name for the next node to be created, usually done by an NVP
        2. Start the node
        3. (if there is data to save) Write the name of the node (this function)
        4. (if there is data to save) Save the data (with saveValue)
        5. Finish the node
        */
        void writeName()
        {
            NodeType const & nodeType = itsNodeStack.top();

            // Start up either an object or an array, depending on state
            if (nodeType == NodeType::StartArray)
            {
                itsWriter.StartArray();
                itsNodeStack.top() = NodeType::InArray;
            }
            else if (nodeType == NodeType::StartObject)
            {
                itsNodeStack.top() = NodeType::InObject;
                itsWriter.StartObject();
            }

            // Array types do not output names
            if (nodeType == NodeType::InArray) return;

            if (itsNextName == nullptr)
            {
                std::string name = "value" + std::to_string(itsNameCounter.top()++) + "\0";
                saveValue(name);
            }
            else
            {
                saveValue(itsNextName);
                itsNextName = nullptr;
            }
        }

        //! Designates that the current node should be output as an array, not an object
        void makeArray()
        {
            itsNodeStack.top() = NodeType::StartArray;
        }

        //! @}

    protected:
    }; // JSONOutputArchive

       // ######################################################################
       //! An input archive designed to load data from JSON
       /*! This archive uses RapidJSON to read in a JSON archive.

       As with the output JSON archive, the preferred way to use this archive is in
       an RAII fashion, ensuring its destruction after all data has been read.

       Input JSON should have been produced by the JSONOutputArchive.  Data can
       only be added to dynamically sized containers (marked by JSON arrays) -
       the input archive will determine their size by looking at the number of child nodes.
       Only JSON originating from a JSONOutputArchive is officially supported, but data
       from other sources may work if properly formatted.

       The JSONInputArchive does not require that nodes are loaded in the same
       order they were saved by JSONOutputArchive.  Using name value pairs (NVPs),
       it is possible to load in an out of order fashion or otherwise skip/select
       specific nodes to load.

       The default behavior of the input archive is to read sequentially starting
       with the first node and exploring its children.  When a given NVP does
       not match the read in name for a node, the archive will search for that
       node at the current level and load it if it exists.  After loading an out of
       order node, the archive will then proceed back to loading sequentially from
       its new position.

       Consider this simple example where loading of some data is skipped:

       @code{cpp}
       // imagine the input file has someData(1-9) saved in order at the top level node
       ar( someData1, someData2, someData3 );        // XML loads in the order it sees in the file
       ar( cereal::make_nvp( "hello", someData6 ) ); // NVP given does not
       // match expected NVP name, so we search
       // for the given NVP and load that value
       ar( someData7, someData8, someData9 );        // with no NVP given, loading resumes at its
       // current location, proceeding sequentially
       @endcode

       \ingroup Archives */
    class JSONInputArchive : public cereal::JSONInputArchive
    {
    protected:
        using ReadStream = rapidjson::IStreamWrapper;
        typedef rapidjson::GenericValue<rapidjson::UTF8<>> JSONValue;
        typedef JSONValue::ConstMemberIterator MemberIterator;
        typedef JSONValue::ConstValueIterator ValueIterator;
        typedef rapidjson::Document::GenericValue GenericValue;

    public:
        std::map<std::string, std::map<std::string, InputInfo>> input_mappings;
        std::map<std::string, std::vector<std::string>> parent_mappings;
        const std::map<std::string, std::string>& variable_replace_mapping;
        const std::map<std::string, std::string>& string_replace_mapping;
        /*! @name Common Functionality
        Common use cases for directly interacting with an JSONInputArchive */
        //! @{

        //! Construct, reading from the provided stream
        /*! @param stream The stream to read from */
        JSONInputArchive(std::istream & stream, const std::map<std::string, std::string>& vm, const std::map<std::string, std::string>& sm) :
            cereal::JSONInputArchive(stream),
            variable_replace_mapping(vm),
            string_replace_mapping(sm)
        {
            
        }

        ~JSONInputArchive() CEREAL_NOEXCEPT = default;

        //! Loads some binary data, encoded as a base64 string
        /*! This will automatically start and finish a node to load the data, and can be called directly by
        users.

        Note that this follows the same ordering rules specified in the class description in regards
        to loading in/out of order */
        void loadBinaryValue(void * data, size_t size, const char * name = nullptr)
        {
            itsNextName = name;

            std::string encoded;
            loadValue(encoded);
            auto decoded = cereal::base64::decode(encoded);

            if (size != decoded.size())
                throw cereal::Exception("Decoded binary data size does not match specified size");

            std::memcpy(data, decoded.data(), decoded.size());
            itsNextName = nullptr;
        };

    

        //! Searches for the expectedName node if it doesn't match the actualName
        /*! This needs to be called before every load or node start occurs.  This function will
        check to see if an NVP has been provided (with setNextName) and if so, see if that name matches the actual
        next name given.  If the names do not match, it will search in the current level of the JSON for that name.
        If the name is not found, an exception will be thrown.

        Resets the NVP name after called.

        @throws Exception if an expectedName is given and not found */
        inline bool search()
        {
            // The name an NVP provided with setNextName()
            if (itsNextName)
            {
                // The actual name of the current node
                auto const actualName = itsIteratorStack.back().name();

                // Do a search if we don't see a name coming up, or if the names don't match
                if (!actualName || std::strcmp(itsNextName, actualName) != 0) {
                    bool nameFound = itsIteratorStack.back().search(itsNextName, itsNextOptional);
                    if (!nameFound && itsNextOptional) {
                        itsLoadOptional = true;
                        itsNextName = nullptr;
                        return false;
                    }else
                    {
                        itsNextName = nullptr;
                        return true;
                    }

                }
            }

            itsNextName = nullptr;
        }

    public:
        //! Starts a new node, going into its proper iterator
        /*! This places an iterator for the next node to be parsed onto the iterator stack.  If the next
        node is an array, this will be a value iterator, otherwise it will be a member iterator.

        By default our strategy is to start with the document root node and then recursively iterate through
        all children in the order they show up in the document.
        We don't need to know NVPs to do this; we'll just blindly load in the order things appear in.

        If we were given an NVP, we will search for it if it does not match our the name of the next node
        that would normally be loaded.  This functionality is provided by search(). */
        void startNode()
        {
            search();

            if (itsIteratorStack.back().value().IsArray())
                itsIteratorStack.emplace_back(itsIteratorStack.back().value().Begin(), itsIteratorStack.back().value().End());
            else
                itsIteratorStack.emplace_back(itsIteratorStack.back().value().MemberBegin(), itsIteratorStack.back().value().MemberEnd());
        }

        //! Finishes the most recently started node
        void finishNode()
        {
            itsIteratorStack.pop_back();
            ++itsIteratorStack.back();
        }

        //! Retrieves the current node name
        /*! @return nullptr if no name exists */
        const char * getNodeName() const
        {
            return itsIteratorStack.back().name();
        }

        //! Sets the name for the next node created with startNode
        void setNext(const char * name, bool optional)
        {
            itsNextName = name;
            itsNextOptional = false;
            itsLoadOptional = optional;
        }
        //! Gets the flag indicating to load optional value
        bool getLoadOptional()
        {
            return itsLoadOptional;
        }

        //! Loads a value from the current node - small signed overload
        template <class T, cereal::traits::EnableIf<std::is_signed<T>::value,
            sizeof(T) < sizeof(int64_t)> = cereal::traits::sfinae> inline
            void loadValue(T & val)
        {
            if (itsNextName)
            {
                auto itr = variable_replace_mapping.find(itsNextName);
                if (itr != variable_replace_mapping.end())
                {
                    val = boost::lexical_cast<T>(itr->second);
                    itsNextName = nullptr;
                    ++itsIteratorStack.back();
                    return;
                }
            }
            search();

            val = static_cast<T>(itsIteratorStack.back().value().GetInt());
            ++itsIteratorStack.back();
        }

        //! Loads a value from the current node - small unsigned overload
        template <class T, cereal::traits::EnableIf<std::is_unsigned<T>::value,
            sizeof(T) < sizeof(uint64_t),
            !std::is_same<bool, T>::value> = cereal::traits::sfinae> inline
            void loadValue(T & val)
        {
            if (itsNextName)
            {
                auto itr = variable_replace_mapping.find(itsNextName);
                if (itr != variable_replace_mapping.end())
                {
                    val = boost::lexical_cast<T>(itr->second);
                    itsNextName = nullptr;
                    ++itsIteratorStack.back();
                    return;
                }
            }
            search();

            val = static_cast<T>(itsIteratorStack.back().value().GetUint());
            ++itsIteratorStack.back();
        }

        //! Loads a value from the current node - bool overload
        void loadValue(bool & val) 
        { 
            if (itsNextName)
            {
                auto itr = variable_replace_mapping.find(itsNextName);
                if (itr != variable_replace_mapping.end())
                {
                    val = boost::lexical_cast<bool>(itr->second);
                    itsNextName = nullptr;
                    ++itsIteratorStack.back();
                    return;
                }
            }
            if(search())
            {
                val = itsIteratorStack.back().value().GetBool();
                ++itsIteratorStack.back();
            }
        }
        //! Loads a value from the current node - int64 overload
        void loadValue(int64_t & val) 
        { 
            if (itsNextName)
            {
                auto itr = variable_replace_mapping.find(itsNextName);
                if (itr != variable_replace_mapping.end())
                {
                    val = boost::lexical_cast<int64_t>(itr->second);
                    itsNextName = nullptr;
                    ++itsIteratorStack.back();
                    return;
                }
            }
            search(); 
            val = itsIteratorStack.back().value().GetInt64(); 
            ++itsIteratorStack.back(); 
        }
        //! Loads a value from the current node - uint64 overload
        void loadValue(uint64_t & val) 
        { 
            if (itsNextName)
            {
                auto itr = variable_replace_mapping.find(itsNextName);
                if (itr != variable_replace_mapping.end())
                {
                    val = boost::lexical_cast<uint64_t>(itr->second);
                    itsNextName = nullptr;
                    ++itsIteratorStack.back();
                    return;
                }
            }
            search(); 
            val = itsIteratorStack.back().value().GetUint64(); 
            ++itsIteratorStack.back(); 
        }
        //! Loads a value from the current node - float overload
        void loadValue(float & val) 
        { 
            if (itsNextName)
            {
                auto itr = variable_replace_mapping.find(itsNextName);
                if (itr != variable_replace_mapping.end())
                {
                    val = boost::lexical_cast<float>(itr->second);
                    itsNextName = nullptr;
                    ++itsIteratorStack.back();
                    return;
                }
            }
            search(); 
            val = static_cast<float>(itsIteratorStack.back().value().GetDouble()); 
            ++itsIteratorStack.back(); 
        }
        //! Loads a value from the current node - double overload
        void loadValue(double & val) 
        { 
            if (itsNextName)
            {
                auto itr = variable_replace_mapping.find(itsNextName);
                if (itr != variable_replace_mapping.end())
                {
                    val = boost::lexical_cast<double>(itr->second);
                    itsNextName = nullptr;
                    ++itsIteratorStack.back();
                    return;
                }
            }
            search(); 
            val = itsIteratorStack.back().value().GetDouble(); 
            ++itsIteratorStack.back(); 
        }
        //! Loads a value from the current node - string overload
        void loadValue(std::string & val) 
        { 
            if (itsNextName)
            {
                auto itr = variable_replace_mapping.find(itsNextName);
                if (itr != variable_replace_mapping.end())
                {
                    val = itr->second;
                    itsNextName = nullptr;
                    ++itsIteratorStack.back();
                    return;
                }
            }
            search();
            val = itsIteratorStack.back().value().GetString();
            
            auto itr1 = string_replace_mapping.find(val);
            if(itr1 != string_replace_mapping.end())
            {
                val = itr1->second;
            }
            ++itsIteratorStack.back();
        }
        //! Loads a nullptr from the current node
        void loadValue(std::nullptr_t&) { search(); CEREAL_RAPIDJSON_ASSERT(itsIteratorStack.back().value().IsNull()); ++itsIteratorStack.back(); }

        // Special cases to handle various flavors of long, which tend to conflict with
        // the int32_t or int64_t on various compiler/OS combinations.  MSVC doesn't need any of this.


    
        
    };

    // ######################################################################
    // JSONArchive prologue and epilogue functions
    // ######################################################################

    // ######################################################################
    //! Prologue for NVPs for JSON archives
    /*! NVPs do not start or finish nodes - they just set up the names */
    template <class T> inline
        void prologue(JSONOutputArchive &, cereal::NameValuePair<T> const &)
    { }

    //! Prologue for NVPs for JSON archives
    template <class T> inline
        void prologue(JSONInputArchive &, cereal::NameValuePair<T> const &)
    { }

    // ######################################################################
    //! Epilogue for NVPs for JSON archives
    /*! NVPs do not start or finish nodes - they just set up the names */
    template <class T> inline
        void epilogue(JSONOutputArchive &, cereal::NameValuePair<T> const &)
    { }

    //! Epilogue for NVPs for JSON archives
    /*! NVPs do not start or finish nodes - they just set up the names */
    template <class T> inline
        void epilogue(JSONInputArchive &, cereal::NameValuePair<T> const &)
    { }

    //! Prologue for NVPs for JSON output archives
    /*! NVPs do not start or finish nodes - they just set up the names */
    template <class T> inline
        void prologue(JSONOutputArchive &, cereal::OptionalNameValuePair<T> const &)
    { }

    //! Prologue for NVPs for JSON input archives
    template <class T> inline
        void prologue(JSONInputArchive &, cereal::OptionalNameValuePair<T> const &)
    { }

    // ######################################################################
    //! Epilogue for NVPs for JSON output archives
    /*! NVPs do not start or finish nodes - they just set up the names */
    template <class T> inline
        void epilogue(JSONOutputArchive &, cereal::OptionalNameValuePair<T> const &)
    { }

    //! Epilogue for NVPs for JSON input archives
    template <class T> inline
        void epilogue(JSONInputArchive &, cereal::OptionalNameValuePair<T> const &)
    { }

    // ######################################################################

    // ######################################################################
    //! Prologue for SizeTags for JSON archives
    /*! SizeTags are strictly ignored for JSON, they just indicate
    that the current node should be made into an array */
    template <class T> inline
        void prologue(JSONOutputArchive & ar, cereal::SizeTag<T> const &)
    {
        ar.makeArray();
    }

    //! Prologue for SizeTags for JSON archives
    template <class T> inline
        void prologue(JSONInputArchive &, cereal::SizeTag<T> const &)
    { }

    // ######################################################################
    //! Epilogue for SizeTags for JSON archives
    /*! SizeTags are strictly ignored for JSON */
    template <class T> inline
        void epilogue(JSONOutputArchive &, cereal::SizeTag<T> const &)
    { }

    //! Epilogue for SizeTags for JSON archives
    template <class T> inline
        void epilogue(JSONInputArchive &, cereal::SizeTag<T> const &)
    { }

    // ######################################################################
    //! Prologue for all other types for JSON archives (except minimal types)
    /*! Starts a new node, named either automatically or by some NVP,
    that may be given data by the type about to be archived

    Minimal types do not start or finish nodes */
    template <class T, cereal::traits::EnableIf<!std::is_arithmetic<T>::value,
        !cereal::traits::has_minimal_base_class_serialization<T, cereal::traits::has_minimal_output_serialization, JSONOutputArchive>::value,
        !cereal::traits::has_minimal_output_serialization<T, JSONOutputArchive>::value> = cereal::traits::sfinae>
        inline void prologue(JSONOutputArchive & ar, T const &)
    {
        ar.startNode();
    }

    //! Prologue for all other types for JSON archives
    template <class T, cereal::traits::EnableIf<!std::is_arithmetic<T>::value,
        !cereal::traits::has_minimal_base_class_serialization<T, cereal::traits::has_minimal_input_serialization, JSONInputArchive>::value,
        !cereal::traits::has_minimal_input_serialization<T, JSONInputArchive>::value> = cereal::traits::sfinae>
        inline void prologue(JSONInputArchive & ar, T const &)
    {
        ar.startNode();
    }

    // ######################################################################
    //! Epilogue for all other types other for JSON archives (except minimal types)
    /*! Finishes the node created in the prologue

    Minimal types do not start or finish nodes */
    template <class T, cereal::traits::EnableIf<!std::is_arithmetic<T>::value,
        !cereal::traits::has_minimal_base_class_serialization<T, cereal::traits::has_minimal_output_serialization, JSONOutputArchive>::value,
        !cereal::traits::has_minimal_output_serialization<T, JSONOutputArchive>::value> = cereal::traits::sfinae>
        inline void epilogue(JSONOutputArchive & ar, T const &)
    {
        ar.finishNode();
    }

    //! Epilogue for all other types other for JSON archives
    template <class T, cereal::traits::EnableIf<!std::is_arithmetic<T>::value,
        !cereal::traits::has_minimal_base_class_serialization<T, cereal::traits::has_minimal_input_serialization, JSONInputArchive>::value,
        !cereal::traits::has_minimal_input_serialization<T, JSONInputArchive>::value> = cereal::traits::sfinae>
        inline void epilogue(JSONInputArchive & ar, T const &)
    {
        ar.finishNode();
    }

    // ######################################################################
    //! Prologue for arithmetic types for JSON archives
    inline
        void prologue(JSONOutputArchive & ar, std::nullptr_t const &)
    {
        ar.writeName();
    }

    //! Prologue for arithmetic types for JSON archives
    inline
        void prologue(JSONInputArchive &, std::nullptr_t const &)
    { }

    // ######################################################################
    //! Epilogue for arithmetic types for JSON archives
    inline
        void epilogue(JSONOutputArchive &, std::nullptr_t const &)
    { }

    //! Epilogue for arithmetic types for JSON archives
    inline
        void epilogue(JSONInputArchive &, std::nullptr_t const &)
    { }

    // ######################################################################
    //! Prologue for arithmetic types for JSON archives
    template <class T, cereal::traits::EnableIf<std::is_arithmetic<T>::value> = cereal::traits::sfinae> inline
        void prologue(JSONOutputArchive & ar, T const &)
    {
        ar.writeName();
    }

    //! Prologue for arithmetic types for JSON archives
    template <class T, cereal::traits::EnableIf<std::is_arithmetic<T>::value> = cereal::traits::sfinae> inline
        void prologue(JSONInputArchive &, T const &)
    { }

    // ######################################################################
    //! Epilogue for arithmetic types for JSON archives
    template <class T, cereal::traits::EnableIf<std::is_arithmetic<T>::value> = cereal::traits::sfinae> inline
        void epilogue(JSONOutputArchive &, T const &)
    { }

    //! Epilogue for arithmetic types for JSON archives
    template <class T, cereal::traits::EnableIf<std::is_arithmetic<T>::value> = cereal::traits::sfinae> inline
        void epilogue(JSONInputArchive &, T const &)
    { }

    // ######################################################################
    //! Prologue for strings for JSON archives
    template<class CharT, class Traits, class Alloc> inline
        void prologue(JSONOutputArchive & ar, std::basic_string<CharT, Traits, Alloc> const &)
    {
        ar.writeName();
    }

    //! Prologue for strings for JSON archives
    template<class CharT, class Traits, class Alloc> inline
        void prologue(JSONInputArchive &, std::basic_string<CharT, Traits, Alloc> const &)
    { }

    // ######################################################################
    //! Epilogue for strings for JSON archives
    template<class CharT, class Traits, class Alloc> inline
        void epilogue(JSONOutputArchive &, std::basic_string<CharT, Traits, Alloc> const &)
    { }

    //! Epilogue for strings for JSON archives
    template<class CharT, class Traits, class Alloc> inline
        void epilogue(JSONInputArchive &, std::basic_string<CharT, Traits, Alloc> const &)
    { }

    // ######################################################################
    // Common JSONArchive serialization functions
    // ######################################################################
    //! Serializing NVP types to JSON
    template <class T> inline
        void CEREAL_SAVE_FUNCTION_NAME(JSONOutputArchive & ar, cereal::NameValuePair<T> const & t)
    {
        ar.setNextName(t.name);
        ar(t.value);
    }

    template <class T> inline
        void CEREAL_LOAD_FUNCTION_NAME(JSONInputArchive & ar, cereal::NameValuePair<T> & t)
    {
        ar.setNext(t.name, false);
        ar(t.value);
    }

    //! Serializing optional NVP types to JSON
    template <class T> inline
        void CEREAL_LOAD_FUNCTION_NAME(JSONInputArchive & ar, cereal::OptionalNameValuePair<T> & t)
    {
        ar.setNext(t.name, true);
        ar(t.value);
        if (ar.getLoadOptional())
        {
            t.value = t.defaultValue;
        }
    }

    //! Saving for nullptr to JSON
    inline
        void CEREAL_SAVE_FUNCTION_NAME(JSONOutputArchive & ar, std::nullptr_t const & t)
    {
        ar.saveValue(t);
    }

    //! Loading arithmetic from JSON
    inline
        void CEREAL_LOAD_FUNCTION_NAME(JSONInputArchive & ar, std::nullptr_t & t)
    {
        ar.loadValue(t);
    }

    //! Saving for arithmetic to JSON
    template <class T, cereal::traits::EnableIf<std::is_arithmetic<T>::value> = cereal::traits::sfinae> inline
        void CEREAL_SAVE_FUNCTION_NAME(JSONOutputArchive & ar, T const & t)
    {
        ar.saveValue(t);
    }

    //! Loading arithmetic from JSON
    template <class T, cereal::traits::EnableIf<std::is_arithmetic<T>::value> = cereal::traits::sfinae> inline
        void CEREAL_LOAD_FUNCTION_NAME(JSONInputArchive & ar, T & t)
    {
        ar.loadValue(t);
    }

    //! saving string to JSON
    template<class CharT, class Traits, class Alloc> inline
        void CEREAL_SAVE_FUNCTION_NAME(JSONOutputArchive & ar, std::basic_string<CharT, Traits, Alloc> const & str)
    {
        ar.saveValue(str);
    }

    //! loading string from JSON
    template<class CharT, class Traits, class Alloc> inline
        void CEREAL_LOAD_FUNCTION_NAME(JSONInputArchive & ar, std::basic_string<CharT, Traits, Alloc> & str)
    {
        ar.loadValue(str);
    }

    // ######################################################################
    //! Saving SizeTags to JSON
    template <class T> inline
        void CEREAL_SAVE_FUNCTION_NAME(JSONOutputArchive &, cereal::SizeTag<T> const &)
    {
        // nothing to do here, we don't explicitly save the size
    }

    //! Loading SizeTags from JSON
    template <class T> inline
        void CEREAL_LOAD_FUNCTION_NAME(JSONInputArchive & ar, cereal::SizeTag<T> & st)
    {
        ar.loadSize(st.size);
    }
} // namespace EagleLib

namespace cereal
{
    inline void save(JSONOutputArchive& ar, rcc::shared_ptr<EagleLib::IDataStream> const & stream)
    {
        auto nodes = stream->GetAllNodes();
        ar(CEREAL_NVP(nodes));
    }
    inline void save(JSONOutputArchive& ar, std::vector<mo::IParameter*> const& parameters)
    {
        for (auto& param : parameters)
        {
            if (param->CheckFlags(mo::Output_e) || param->CheckFlags(mo::Input_e))
                continue;
            auto func1 = mo::SerializationFunctionRegistry::Instance()->GetJsonSerializationFunction(param->GetTypeInfo());
            if (func1)
            {
                if (!func1(param, ar))
                {
                    LOG(debug) << "Unable to deserialize " << param->GetName() << " of type " << param->GetTypeInfo().name();
                }
            }
            else
            {
                LOG(debug) << "No serialization function exists for  " << param->GetName() << " of type " << param->GetTypeInfo().name();
            }
        }
    }


    inline void save(JSONOutputArchive& ar, std::vector<mo::InputParameter*> const& parameters)
    {
        for (auto& param : parameters)
        {

            mo::InputParameter* input_param = dynamic_cast<mo::InputParameter*>(param);
            EagleLib::InputInfo info;
            info.type = "Direct";
            if (input_param)
            {
                mo::IParameter* _input_param = input_param->GetInputParam();
                std::string input_name;
                if (_input_param)
                {
                    input_name = _input_param->GetTreeName();
                    auto pos = input_name.find(" buffer for ");
                    if(pos != std::string::npos)
                    {
                        input_name = input_name.substr(0, input_name.find_first_of(' '));
                        //input_name = input_name.substr(0, pos);
                    }

                    if(_input_param->CheckFlags(mo::Buffer_e))
                    {
                        mo::Buffer::IBuffer* buffer = dynamic_cast<mo::Buffer::IBuffer*>(_input_param);
                        if(buffer)
                        {
                            info.type = mo::ParameterTypeFlagsToString(buffer->GetBufferType());
                        }
                    }
                    info.name = input_name;
                }
            }
            ar(cereal::make_nvp(param->GetName(), info));
        }
    }
    inline void save(JSONOutputArchive& ar, rcc::weak_ptr<EagleLib::Nodes::Node> const& node)
    {
        std::string name = node->GetTreeName();
        ar(CEREAL_NVP(name));
    }
    inline void save(JSONOutputArchive& ar, rcc::weak_ptr<EagleLib::Algorithm> const& obj)
    {
        auto parameters = obj->GetParameters();
        std::string type = obj->GetTypeName();
        const auto& components = obj->GetComponents();
        ar(CEREAL_NVP(type));
        ar(CEREAL_NVP(parameters));
        ar(CEREAL_NVP(components));
    }

    inline void save(JSONOutputArchive& ar, rcc::shared_ptr<EagleLib::Nodes::Node> const& node)
    {
        auto parameters = node->GetParameters();
        std::string type = node->GetTypeName();
        std::string name = node->GetTreeName();
        const auto& components = node->GetComponents();
        ar(CEREAL_NVP(type));
        ar(CEREAL_NVP(name));
        ar(CEREAL_NVP(parameters));
        ar(CEREAL_NVP(components));
        auto inputs = node->GetInputs();
        ar(CEREAL_NVP(inputs));
        auto parent_nodes = node->GetParents();
        std::vector<std::string> parents;
        for(auto& node : parent_nodes)
        {
            parents.emplace_back(node->GetTreeName());
        }
        ar(CEREAL_NVP(parents));
    }


    inline void load(JSONInputArchive& ar, rcc::shared_ptr<EagleLib::IDataStream>& stream)
    {
        if(stream == nullptr)
        {
        }
        std::vector<rcc::shared_ptr<EagleLib::Nodes::Node>> nodes;
        ar(CEREAL_NVP(nodes));
        EagleLib::JSONInputArchive& ar_ = dynamic_cast<EagleLib::JSONInputArchive&>(ar);
        for(int i = 0; i < nodes.size(); ++i)
        {
            if(nodes[0] == nullptr)
                continue;
            nodes[i]->SetDataStream(stream.Get());
            nodes[i]->PostSerializeInit();
            auto& parents = ar_.parent_mappings[nodes[i]->GetTreeName()];
            for (auto& parent : parents)
            {
                for(int j = 0; j < nodes.size(); ++j)
                {
                    if(i != j)
                    {
                        if(nodes[j]->GetTreeName() == parent)
                        {
                            nodes[j]->AddChild(nodes[i]);
                        }
                    }
                }
            }
            auto& input_mappings = ar_.input_mappings[nodes[i]->GetTreeName()];
            auto input_params = nodes[i]->GetInputs();
            for(auto& input : input_params)
            {

                auto itr = input_mappings.find(input->GetName());
                if(itr != input_mappings.end())
                {
                       auto pos = itr->second.name.find(":");
                       if(pos != std::string::npos)
                       {
                           std::string output_node_name = itr->second.name.substr(0, pos);
                           for(int j = 0; j < nodes.size(); ++j)
                           {
                                if(nodes[j]->GetTreeName() == output_node_name)
                                {
                                    auto space_pos = itr->second.name.find(' ');
                                    auto output_param = nodes[j]->GetOutput(itr->second.name.substr(pos + 1, space_pos - (pos + 1)));
                                    if (!output_param)
                                    {
                                        LOG(warning) << "Unable to find parameter " << itr->second.name.substr(pos + 1) << " in node " << nodes[j]->GetTreeName();
                                        break;
                                    }

                                    std::string type = itr->second.type;
                                    if(type != "Direct")
                                    {
                                          mo::ParameterTypeFlags buffer_type = mo::StringToParameterTypeFlags(type);
                                          if (!nodes[i]->ConnectInput(nodes[j], output_param, input, mo::ParameterTypeFlags(buffer_type | mo::ForceBufferedConnection_e)))
                                          {
                                              LOG(warning) << "Unable to connect " << output_param->GetTreeName() << " (" << output_param->GetTypeInfo().name() << ") to "
                                                  << input->GetTreeName() << " (" << input->GetTypeInfo().name() << ")";
                                          }else
                                          {
                                              if(itr->second.buffer_size > 0)
                                              {
                                                  mo::IParameter* p = input->GetInputParam();
                                                  if(mo::Buffer::IBuffer* b = dynamic_cast<mo::Buffer::IBuffer*>(p))
                                                  {
                                                      b->SetSize(itr->second.buffer_size);
                                                  }
                                              }
                                              if(itr->second.sync)
                                              {
                                                  nodes[i]->SetSyncInput(input->GetName());
                                              }
                                          }
                                    }else
                                    {
                                        if (!nodes[i]->ConnectInput(nodes[j], output_param, input))
                                        {
                                            LOG(warning) << "Unable to connect " << output_param->GetTreeName() << " (" << output_param->GetTypeInfo().name() << ") to "
                                                << input->GetTreeName() << " (" << input->GetTypeInfo().name() << ")";
                                        }else
                                        {
                                           if(itr->second.sync)
                                           {
                                               nodes[i]->SetSyncInput(input->GetName());
                                               LOG(info) << "Node (" << nodes[i]->GetTreeName() << ") syncs to " << input->GetName();
                                           }
                                        }
                                    }
                                }
                           }
                       }else
                       {
                           if(itr->second.name.size())
                               LOG(warning) << "Invalid input format for input [" << itr->second.name << "] of node: " << nodes[i]->GetTreeName();
                       }
                }else
                {
                    if(input->CheckFlags(mo::Optional_e))
                    {
                        LOG(debug) << "Unable to find input setting for " << input->GetName() << " for node " << nodes[i]->GetTreeName();
                    }else
                    {
                        LOG(warning) << "Unable to find input setting for " << input->GetName() << " for node " << nodes[i]->GetTreeName();
                    }
                }
            }
        }
        for(int i = 0; i < nodes.size(); ++i)
        {
            if(nodes[i] != nullptr && nodes[i]->GetParents().size() == 0)
            {
                stream->AddNode(nodes[i]);
            }
        }
    }

    inline void load(JSONInputArchive& ar, std::vector<mo::IParameter*>& parameters)
    {
        for (auto& param : parameters)
        {
            if (param->CheckFlags(mo::Output_e) || param->CheckFlags(mo::Input_e))
                continue;
            auto func1 = mo::SerializationFunctionRegistry::Instance()->GetJsonDeSerializationFunction(param->GetTypeInfo());
            if (func1)
            {
                if (!func1(param, ar))
                {
                    LOG(debug) << "Unable to deserialize " << param->GetName() << " of type " << param->GetTypeInfo().name();
                }
            }
            else
            {
                LOG(debug) << "No deserialization function exists for  " << param->GetName() << " of type " << param->GetTypeInfo().name();
            }
        }
    }

    inline void load(JSONInputArchive& ar, std::vector<mo::InputParameter*> & parameters)
    {
        for (auto& param : parameters)
        {
            std::string name = param->GetName();
            EagleLib::InputInfo info;
            auto nvp = cereal::make_optional_nvp(name, info, info);
            ar(nvp);
            if(nvp.success == false)
                return;
            EagleLib::JSONInputArchive& ar_ = dynamic_cast<EagleLib::JSONInputArchive&>(ar);
            ar_.input_mappings[param->GetTreeRoot()][name] = info;
        }
    }
   inline void load(JSONInputArchive& ar, rcc::weak_ptr<EagleLib::Algorithm>& obj)
   {
       std::string type;
       ar(CEREAL_NVP(type));
       if(!obj)
       {
            IObject* ptr =  mo::MetaObjectFactory::Instance()->Create(type.c_str());
            if(ptr)
            {
                EagleLib::Algorithm* alg_ptr = dynamic_cast<EagleLib::Algorithm*>(ptr);
                if(!alg_ptr)
                    delete ptr;
                else
                    obj = alg_ptr;
            }
       }
       if(!obj)
       {
            LOG(warning) << "Unable to create algorithm of type: " << type;
            return;
       }
       auto parameters = obj->GetParameters();
       if(parameters.size())
          ar(CEREAL_NVP(parameters));
       std::vector<rcc::weak_ptr<EagleLib::Algorithm>> components;
       try
       {
           ar(CEREAL_OPTIONAL_NVP(components, components));
       }catch(...)
       {

       }

       if(components.size())
            for(auto component : components)
                obj->AddComponent(component);
   }
    inline void load(JSONInputArchive& ar, rcc::shared_ptr<EagleLib::Nodes::Node>& node)
    {
        std::string type;
        std::string name;
        ar(CEREAL_NVP(type));
        if(!node)
            node = mo::MetaObjectFactory::Instance()->Create(type.c_str());
        ar(CEREAL_NVP(name));
        std::vector<rcc::weak_ptr<EagleLib::Algorithm>> components;
        auto components_nvp = CEREAL_OPTIONAL_NVP(components, components);
        ar(components_nvp);
        if(components_nvp.success)
        {
            for(auto component : components)
            {
                if(component)
                    node->AddComponent(component);
            }
        }

        if (!node)
        {
            LOG(warning) << "Unable to create node with type: " << type;
            return;
        }
        node->SetTreeName(name);
        auto parameters = node->GetParameters();
        for(auto itr = parameters.begin(); itr != parameters.end(); )
        {
            if((*itr)->CheckFlags(mo::Input_e))
            {
                itr = parameters.erase(itr);
            }else
            {
                ++itr;
            }
        }
        if(parameters.size())
            ar(CEREAL_OPTIONAL_NVP(parameters, parameters));
        auto inputs = node->GetInputs();
        if(inputs.size())
            ar(CEREAL_OPTIONAL_NVP(inputs, inputs));
        EagleLib::JSONInputArchive& ar_ = dynamic_cast<EagleLib::JSONInputArchive&>(ar);
        ar(cereal::make_optional_nvp("parents", ar_.parent_mappings[name]));
    }
}

  // register archives for polymorphic support
CEREAL_REGISTER_ARCHIVE(EagleLib::JSONInputArchive)
CEREAL_REGISTER_ARCHIVE(EagleLib::JSONOutputArchive)

// tie input and output archives together
CEREAL_SETUP_ARCHIVE_TRAITS(EagleLib::JSONInputArchive, EagleLib::JSONOutputArchive)


