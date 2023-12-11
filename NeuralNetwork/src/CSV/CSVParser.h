#pragma once
#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>
#include <string>


namespace nn
{
    class CSVParser
    {
    private:
        double m_target;
        bool m_has_header;
        std::vector<double> m_values;
        std::ifstream m_file;

    public:
        CSVParser(std::string filename, bool has_header = false);
        bool endOfFile();
        bool getDataFromSingleLine();
        bool getDataAt(int index);
        double getTarget() const;
        const std::vector<double>& getValues() const;
        void restartFile();
        int countLines();
        ~CSVParser();
    };
}
