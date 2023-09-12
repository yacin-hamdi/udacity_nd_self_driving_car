#include<iostream>
#include "help_functions.h"

float value = 1.0;

float parameter = 1.0;
float stdev = 1.0;

int main() {
	float prob = Helpers::normpdf(value, parameter, stdev);
	std::cout << prob << std::endl;

	return 0;
}