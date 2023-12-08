#pragma once
#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>
#include <string>


class CSVParser
{
private:
    double target;
    bool _hasHeader;
    std::vector<double> values;
    std::ifstream file;

public:
    CSVParser(std::string filename, bool hasHeader = false);
    bool endOfFile();
    bool getDataFromSingleLine();
    bool getDataAt(int index);
    double getTarget() const;
    const std::vector<double>& getValues() const;
    void restartFile();
    int countLines();
    ~CSVParser();
};
