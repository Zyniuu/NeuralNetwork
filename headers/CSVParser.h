#pragma once
#include <vector>
#include <fstream>
#include <sstream>

template <typename T>
class CSVParser
{
private:
    T target;
    std::vector<T> values;
    std::ifstream file;

public:
    CSVParser(std::string filename, bool hasHeader);
    bool endOfFile();
    void getDataFromSingleLine();
    T getTarget() const;
    const std::vector<T>& getValues() const;
    ~CSVParser();
};