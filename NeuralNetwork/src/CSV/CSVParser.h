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
        bool m_is_target_first;
        char m_separator;
        int m_start_column_index;
        std::vector<double> m_values;
        std::ifstream m_file;

    public:
        CSVParser(std::string filename, char separator, int start_column_index = 0, bool has_header = false, bool is_target_first = true);
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
