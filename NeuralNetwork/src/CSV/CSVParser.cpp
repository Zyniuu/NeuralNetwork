#include "CSVParser.h"


namespace nn
{
    CSVParser::CSVParser(std::string filename, char separator, int start_column_index, bool has_header, bool is_target_first)
        : m_separator(separator), m_start_column_index(start_column_index), m_has_header(has_header), m_is_target_first(is_target_first)
    {
        m_file.open(filename);
        // check if file has been opened
        if (!m_file.is_open())
        {
            std::cerr << "Error: Unable to open the file: " << filename << std::endl;
            exit(EXIT_FAILURE);
        }
        // if the file contains a header then skip it
        if (m_has_header)
        {
            std::string line;
            std::getline(m_file, line);
        }
    }


    CSVParser::~CSVParser()
    {
        if (m_file.is_open())
        {
            m_file.close();
        }
        m_values.clear();
    }


    bool CSVParser::endOfFile()
    {
        return (m_file.peek() == EOF);
    }


    double CSVParser::getTarget() const
    {
        return m_target;
    }


    const std::vector<double>& CSVParser::getValues() const
    {
        return m_values;
    }


    void CSVParser::restartFile()
    {
        if (m_file.is_open())
        {
            m_file.clear();
            m_file.seekg(0, std::ios::beg);
            if (m_has_header)
            {
                std::string line;
                std::getline(m_file, line);
            }
        }
    }


    bool CSVParser::getDataFromSingleLine()
    {
        if (m_file.is_open())
        {
            std::string line;
            if (std::getline(m_file, line))
            {
                m_values.clear();

                std::istringstream ss(line);
                std::string cell;
                std::vector<double> cells;
                int current_column_index = 0;

                while (std::getline(ss, cell, m_separator))
                {
                    if (current_column_index >= m_start_column_index)
                    {
                        cells.push_back(std::stod(cell));
                    }
                    ++current_column_index;
                }
                if (!cells.empty())
                {
                    if (m_is_target_first)
                    {
                        m_target = cells.front();
                        cells.erase(cells.begin());
                    }
                    else
                    {
                        m_target = cells.back();
                        cells.pop_back();
                    }
                }
                m_values = cells;
                return true;
            }
            return false;
        }
        return false;
    }


    bool CSVParser::getDataAt(int index)
    {
        if (m_file.is_open())
        {
            for (int i = 0; i <= index; i++)
            {
                if (!getDataFromSingleLine())
                {
                    restartFile();
                    return false;
                }
            }
        }
        restartFile();
        return true;
    }


    int CSVParser::countLines()
    {
        int lines = std::count(std::istreambuf_iterator<char>(m_file), std::istreambuf_iterator<char>(), '\n');
        if (m_has_header)
            lines -= 1;
        restartFile();
        return lines;
    }
}
