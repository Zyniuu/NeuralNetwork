#include "../headers/CSVParser.h"

template <typename T>
CSVParser<T>::CSVParser(std::string filename, bool hasHeader)
{
    file.open(filename);
    std::string line;
    // check if file has been opened
    if (!file.is_open())
    {
        exit(3);
    }
    // if the file contains a header then skip it
    if (hasHeader)
    {
        getline(file, line);
    }
}

template <typename T>
CSVParser<T>::~CSVParser()
{
    if (file.is_open())
    {
        file.close();
    }
}

template <typename T>
bool CSVParser<T>::endOfFile()
{
    return (file.peek() == EOF);
}

template <typename T>
T CSVParser<T>::getTarget() const
{
    return target;
}

template <typename T>
const std::vector<T>& CSVParser<T>::getValues() const
{
    return values;
}

template <typename T>
void CSVParser<T>::restartFile()
{
    file.clear();
    file.seekg(0, std::ios::beg);
}

template <typename T>
void CSVParser<T>::getDataFromSingleLine()
{
    if (!file.good())
    {
        return;
    }
    values.clear();

    std::string line;
    getline(file, line);
    std::stringstream ss(line);

    // treat the first element as a target
    ss >> target;
    if (ss.peek() == ',' || ss.peek() == ' ')
        ss.ignore();

    // rest of the values insert into vector
    for (T i; ss >> i;)
    {
        values.push_back(i);
        if (ss.peek() == ',' || ss.peek() == ' ')
            ss.ignore();
    }
    ss.clear();
}

// resolve linking errors
template class CSVParser<int>;
template class CSVParser<double>;
template class CSVParser<float>;
