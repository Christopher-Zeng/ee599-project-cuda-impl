#include <fstream>
#include <sstream>
#include <stdexcept>

void print_matrix(float *mat, int rows, int cols);

void init_matrix(float *mat, int rows, int cols, float seed);

void read_data(std::string path, int *dim, float *input, float *kernel, float *goldenOutput);

template <typename T>
void read_file(std::ifstream &fs, T *buffer);
