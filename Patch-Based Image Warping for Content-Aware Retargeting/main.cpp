#include <string>

#include "application.h"

int main(int argc, char **argv) {

  std::string input_file_name = "butterfly.jpg";

  if (argc == 2) {
    input_file_name = std::string(argv[1]);
  }

  Application application(input_file_name);

  application.Run();

  return 0;
}