#include "CSVParser.h"


namespace nn
{
    CSVParser::CSVParser(std::string filename, bool has_header)
        : m_has_header(has_header)
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
                bool first_cell = true;

                while (std::getline(ss, cell, ','))
                {
                    if (first_cell)
                    {
                        m_target = std::stod(cell);
                        first_cell = false;
                        continue;
                    }
                    else
                        m_values.push_back(std::stod(cell));
                }
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
