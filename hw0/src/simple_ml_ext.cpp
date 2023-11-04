#include <cstddef>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>

namespace py = pybind11;

void dot(const float *X,
         const float *y,
         float *out,
         size_t m,
         size_t n,
         size_t k) {
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < k; ++j) {
            out[i * k + j] = 0;
            for (size_t x = 0; x < n; ++x) {
                out[i * k + j] += X[i * n + x] * y[x * k + j];
            }
        }
    }
}

void transpose(float *in, float *out, size_t m, size_t n) {
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            out[j * m + i] = in[i * n + j];
        }
    }
}

void multiply(float *X, float k, size_t m, size_t n) {
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            X[i * n + j] *= k;
        }
    }   
}

void softmax(float *in, size_t m, size_t n) {
    for (size_t i = 0; i < m; ++i) {
        float sum = 0;
        for (size_t j = 0; j < n; ++j) {
            in[i * n + j] = std::exp(in[i * n + j]);
            sum += in[i * n + j];
        }
        for (size_t j = 0; j < n; ++j) {
            in[i * n + j] /= sum;
        }
    }
}

template <typename T>
void copy_arr(const T *in, T *out, size_t offset, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        out[i] = in[offset + i];
    }
}

void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
								  float *theta, size_t m, size_t n, size_t k,
								  float lr, size_t batch)
{
    /**
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): pointer to X data, of size m*n, stored in row
     *          major (C) format
     *     y (const unsigned char *): pointer to y data, of size m
     *     theta (float *): pointer to theta data, of size n*k, stored in row
     *          major (C) format
     *     m (size_t): number of examples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (int): SGD minibatch size
     *
     * Returns:
     *     (None)
     */

    /// BEGIN YOUR CODE
    multiply(theta, .0f, n, k);

    float *X_batch = new float[batch * n];
    unsigned char *y_batch = new unsigned char[batch];
    float *Z = new float[batch * k];
    float *X_batch_T = new float[batch * n];
    float *d_theta = new float[n * k];
    for (size_t offset = 0; offset < m; offset += batch) {
        copy_arr<float>(X, X_batch, offset * n, batch * n);
        copy_arr<unsigned char>(y, y_batch, offset, batch);
        dot(X_batch, theta, Z, batch, n, k);
        softmax(Z, batch, k);
        for (size_t i = 0; i < batch; ++i) {
            Z[i * k + y_batch[i]] -= 1;
        }
        transpose(X_batch, X_batch_T, batch, n);
        dot(X_batch_T, Z, d_theta, n, batch, k);
        multiply(d_theta, lr / batch, n, k);
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < k; ++j) {
                theta[i * k + j] -= d_theta[i * k + j];
            }
        }
    }

    delete[] X_batch;
    delete[] y_batch;
    delete[] Z;
    delete[] X_batch_T;
    delete[] d_theta;
    /// END YOUR CODE
}


/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m) {
    m.def("softmax_regression_epoch_cpp",
    	[](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch) {
        softmax_regression_epoch_cpp(
        	static_cast<const float*>(X.request().ptr),
            static_cast<const unsigned char*>(y.request().ptr),
            static_cast<float*>(theta.request().ptr),
            X.request().shape[0],
            X.request().shape[1],
            theta.request().shape[1],
            lr,
            batch
           );
    },
    py::arg("X"), py::arg("y"), py::arg("theta"),
    py::arg("lr"), py::arg("batch"));
}
