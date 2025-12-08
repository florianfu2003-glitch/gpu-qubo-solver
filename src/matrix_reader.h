#pragma once

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <algorithm>
#include <stdexcept>
#include "matrix.h"

template<typename iT, typename vT>
struct Coordinate{
	iT row;
	iT col;
	vT value;
};

template<typename vT, typename iT>
SparseMatrix<vT, iT> readMatrixMarket(const std::string& file_path) {
	std::ifstream file(file_path);
	if (!file.is_open()) {
		throw std::runtime_error("Could not open file: " + file_path);
	}

	std::string line;
    bool symmetric = false;
    bool first_line = true;
	// Skip comment lines
	while (std::getline(file, line)) {
        // Check for symmetry in header
        if (line.find("symmetric") != std::string::npos && first_line) {
            symmetric = true;
        }
        first_line = false;
		if (!line.empty() && (line[0] != '#' && line[0] != '%')) {
			break;
		}
	}
	// Remove potential carriage return character
	if(!line.empty() && line.back() == '\r')
        line.pop_back();

	std::istringstream header_stream(line);
	iT n; //rows
	iT nnz; //nnz
	iT m; //cols

	// read first two values as matrix dimensions
	header_stream >> n >> m; 
	// read third value as number of non-zeros if present, otherwise matrix is square
	if (header_stream.rdbuf()->in_avail())
		header_stream >> nnz;
	else
	{
		nnz = m;
		m = n;
	}

	// store coordinates temporarily
	std::vector<Coordinate<iT, vT>> coo_coordinates;

	//read coordinates
	while (std::getline(file, line)) {
		if (line.empty()) continue;
		if(!line.empty() && line.back() == '\r')
        	line.pop_back();
		std::istringstream iss(line);
		int row, col;
		float val; // stored as float in file, convert to vT
		iss >> row >> col >> val;
        if(symmetric && row != col) {
            // Store symmetric entry
            coo_coordinates.push_back({static_cast<iT>(col - 1), static_cast<iT>(row - 1), static_cast<vT>(val)});
        }
		coo_coordinates.push_back({static_cast<iT>(row - 1), static_cast<iT>(col - 1), static_cast<vT>(val)});
	}
	if (symmetric)
		nnz = coo_coordinates.size();

	//sort coordinates by row and then by column
	std::sort(coo_coordinates.begin(), coo_coordinates.end(), [](const Coordinate<iT, vT>& a, const Coordinate<iT, vT>& b) {
		if (a.row == b.row)
			return a.col < b.col;
		return a.row < b.row;
	});

	file.close();
	
	// Convert to CSR format
    SparseMatrix<vT, iT> mat(n, m);
    mat.allocate(nnz);
    mat.offsets[0] = 0;
	// Count non-zeros per row
	int i = 0;
	for (const auto& coord : coo_coordinates) {
		mat.values[i] = coord.value;
		mat.columns[i] = coord.col;
		i++;
		mat.offsets[coord.row + 1] = i;
	}

	return mat;
}