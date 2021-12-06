#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

void print_matrix(float *mat, int rows, int cols);

void init_matrix(float *mat, int rows, int cols, float seed);

void read_matrix(std::string filename, float *matrix);
