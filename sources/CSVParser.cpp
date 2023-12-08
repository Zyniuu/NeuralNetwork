#include "../headers/CSVParser.h"


CSVParser::CSVParser(std::string filename, bool hasHeader)
{
    _hasHeader = hasHeader;
    file.open(filename);
    // check if file has been opened
    if (!file.is_open())
    {
        std::cerr << "Error: Unable to open the file: " << filename << std::endl;
        exit(EXIT_FAILURE);
    }
    // if the file contains a header then skip it
    if (hasHeader)
    {
        std::string line;
        getline(file, line);
    }
}

CSVParser::~CSVParser()
{
    if (file.is_open())
    {
        file.close();
    }
}

bool CSVParser::endOfFile()
{
    return (file.peek() == EOF);
}

double CSVParser::getTarget() const
{
    return target;
}

const std::vector<double>& CSVParser::getValues() const
{
    return values;
}

void CSVParser::restartFile()
{
    if (file.is_open())
    {
        file.clear();
        file.seekg(0, std::ios::beg);
        if (_hasHeader)
        {
            std::string line;
            std::getline(file, line);
        }
    }
}

bool CSVParser::getDataFromSingleLine()
{
    if (file.is_open())
    {
        std::string line;
        if (std::getline(file, line))
        {
            values.clear();
            
            std::istringstream ss(line);
            std::string cell;
            bool firstCell = true;

            while (std::getline(ss, cell, ','))
            {
                if (firstCell)
                {
                    target = std::stod(cell);
                    firstCell = false;
                    continue;
                }
                else
                    values.push_back(std::stod(cell));
            }
            return true;
        }
        return false;
    }
    return false;
}

bool CSVParser::getDataAt(int index)
{
    if (file.is_open())
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
    int lines = std::count(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>(), '\n');
    if (_hasHeader)
        lines -= 1;
    restartFile();
    return lines;
}
