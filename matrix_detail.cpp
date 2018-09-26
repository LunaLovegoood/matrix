// Matrix
// Copyright (C) 2018 Yurii Khomiak 
// Yurii Khomiak licenses this file to you under the MIT license. 
// See the LICENSE file in the project root for more information.

#include "matrix_detail.h"

#include <iostream>


namespace matrix {

	std::ostream& operator<<(std::ostream &stream, const MatrixDetail &details) {

		stream << details.is_init_ << " " << details.is_square_ << " " << details.is_triangular_;

		return stream;
	}

	std::istream& operator>>(std::istream &stream, MatrixDetail &details) {

		stream >> details.is_init_ >> details.is_square_ >> details.is_triangular_;

		return stream;
	}

}
