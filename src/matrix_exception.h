// Matrix
// Copyright (C) 2018 Yurii Khomiak 
// Yurii Khomiak licenses this file to you under the MIT license. 
// See the LICENSE file in the project root for more information.

#ifndef MATRIX_EXCEPTION_H_
#define MATRIX_EXCEPTION_H_

#include <string>


namespace matrix {

    // Base class for matrix exceptions
    class MatrixExceptions {
    protected:
        std::string message_{};
    public:
        MatrixExceptions(const std::string &message) : message_{ message } {}
        virtual const std::string& what() const = 0;
        virtual ~MatrixExceptions() {}
    };

    // Exception for incorrect dimensions for performing certain matrix operations
    class IncorrectDimensions : public MatrixExceptions {
    public:
        IncorrectDimensions() : MatrixExceptions{ "Incorrect matrix dimensions for this operation.\n" } {}
        const std::string& what() const override { return message_; };
    };

    // Exception for uninitialized matrix
    class UninitializedMatrix : public MatrixExceptions {
    public:
        UninitializedMatrix() : MatrixExceptions{ "Uninitialized matrix.\n" } {}
        const std::string& what() const override { return message_; };
    };

    // Exception for trying to reach out of bounds matrix element
    class MatrixOutOfBounds : public MatrixExceptions {
    public:
        MatrixOutOfBounds() : MatrixExceptions{ "Matrix indexes out of bounds.\n" } {}
        const std::string& what() const override { return message_; };
    };

    // Exception for calculating the determinant for non-square matrices
    class DetDoesNotExist : public MatrixExceptions {
    public:
        DetDoesNotExist() : MatrixExceptions{ "Determinant does not exist for given matrix.\n" } {}
        const std::string& what() const override { return message_; };
    };

    // Exception for situations where inverse does not exist
    class InverseDoesNotExist : public MatrixExceptions {
    public:
        InverseDoesNotExist() : MatrixExceptions{ "Inverse does not exist for given matrix.\n" } {}
        const std::string& what() const override { return message_; };
    };

    // Exception for situation when lower bound for random is greater than upper
    class IncorrectBoundsForRandom : public MatrixExceptions {
    public:
        IncorrectBoundsForRandom() : MatrixExceptions{ "Lower bound is greater than upper bound.\n" } {}
        const std::string& what() const override { return message_; };
    };

}

#endif // MATRIX_EXCEPTION_H_
