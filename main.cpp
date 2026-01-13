#include <iostream>
#include <vector>
#include <random>
#include <cmath>

using namespace std;

class TensorTransform {
public:
    virtual double apply(double x) const = 0;
    virtual ~TensorTransform() = default;
};

class ReLU : public TensorTransform {
public:
    double apply(double x) const override {
        return (x > 0.0) ? x : 0.0;
    }
};

class Sigmoid : public TensorTransform {
public:
    double apply(double x) const override {
        if (x >= 0.0) {
            double z = std::exp(-x);
            return 1.0 / (1.0 + z);
        } else {
            double z = std::exp(x);
            return z / (1.0 + z);
        }
    }
};

class Tensor {

    friend Tensor dot(const Tensor& a, const Tensor& b);
    friend Tensor matmul(const Tensor& a, const Tensor& b);

    size_t shape;
    size_t shape_[3];
    double* values;
    size_t* ref;

    Tensor(size_t shape, const size_t shape_[3], double* values, size_t* ref) {
        this->shape = shape;
        this->values = values;
        this->ref = ref;

        for (size_t i = 0; i < shape; ++i)
            this->shape_[i] = shape_[i];
        ++(*ref);
    }

public:

    Tensor() = default;
    Tensor(const vector<size_t>& shape, const vector<double>& values) {
        this->shape = shape.size();
        this->values = nullptr;
        this->ref = nullptr;

        if (this->shape < 1 || this->shape > 3 ) {
            throw invalid_argument("shape must be between 1 and 3");
        }

        size_t total_size = 1;
        for (size_t i = 0; i < this->shape; ++i) {
            if (shape[i] == 0) {
                throw invalid_argument("shape could not be 0");
            }
            shape_[i] = shape[i];
            total_size *= shape[i];
        }



        if (values.size() != total_size) {
            throw invalid_argument("shape must have the same number of elements");
        }

        this->values = new double[total_size];

        for (size_t i = 0; i < total_size; i++) {
            this->values[i] = values[i];
        }

        ref = new size_t(1);
    }
    Tensor(const Tensor& other) {
        this->shape = other.shape;
        this->values = other.values;
        this->ref = other.ref;

        for (size_t i = 0; i < shape; ++i)
            shape_[i] = other.shape_[i];
        ++(*ref);
    }
    Tensor& operator=(const Tensor& other) {
        if (this == &other) return *this;

        if (ref) {
            if (--(*ref) == 0) {
                delete[] values;
                delete ref;
            }
        }

        shape = other.shape;
        for (size_t i = 0; i < shape; ++i)
            shape_[i] = other.shape_[i];
        values = other.values;
        ref = other.ref;
        ++(*ref);

        return *this;
    }
    Tensor(Tensor&& other) {
        this->shape = other.shape;
        this->values = other.values;
        this->ref = other.ref;

        for (size_t i = 0; i < shape; ++i)
            shape_[i] = other.shape_[i];
        other.values = nullptr;
        other.ref = nullptr;
        other.shape = 0;
    }
    Tensor& operator=(Tensor&& other) {
        if (this == &other) return *this;

        if (ref) {
            if (--(*ref) == 0) {
                delete[] values;
                delete ref;
            }
        }

        shape = other.shape;
        for (size_t i = 0; i < shape; ++i)
            shape_[i] = other.shape_[i];
        values = other.values;
        ref = other.ref;

        other.values = nullptr;
        other.ref = nullptr;
        other.shape = 0;

        return *this;
    }

    ~Tensor() {
        if (!ref) return;
        if (--(*ref) == 0) {
            delete[] values;
            delete ref;
        }
    }

    size_t size() {
        size_t total = 1;
        for (size_t i = 0; i < shape; ++i) total *= shape_[i];
        return total;
    }

    static Tensor zeros(const vector<size_t>& shape) {
        size_t total_size = 1;
        for (size_t i = 0; i < shape.size(); ++i) {
            total_size *= shape[i];
        }

        vector<double> values(total_size, 0.0);
        return Tensor(shape, values);
    }
    static Tensor ones(const vector<size_t>& shape) {
        size_t total_size = 1;
        for (size_t i = 0; i < shape.size(); ++i) {
            total_size *= shape[i];
        }

        vector<double> values(total_size, 1.0);
        return Tensor(shape, values);
    }
    static Tensor random(const vector<size_t>& shape, double min, double max) {
        size_t total_size = 1;
        for (size_t i = 0; i < shape.size(); ++i) {
            total_size *= shape[i];
        }

        random_device rd;
        mt19937 gen(rd());
        uniform_real_distribution<double> dis(min, max);

        vector<double> values(total_size);
        for (size_t i = 0; i < total_size; ++i) {
            values[i] = dis(gen);
        }

        return Tensor(shape, values);
    }
    static Tensor arange(long long start, long long end) {
        if (start >= end) {
            throw invalid_argument("start must be less than end");
        }

        size_t n = static_cast<size_t>(end - start);

        vector<size_t> shape = { n };
        vector<double> values(n);

        for (size_t i = 0; i < n; ++i) {
            values[i] = static_cast<double>(start + static_cast<long long>(i));
        }

        return Tensor(shape, values);
    }

    Tensor operator+(const Tensor& other) {
        if (shape != other.shape) {
            throw invalid_argument("shape must have the same number of elements");
        }

        size_t total_size = 1;
        for (size_t i = 0; i < shape; ++i) {
            total_size *= shape_[i];
        }

        vector<double> newValues(total_size);
        for (size_t i = 0; i < total_size; ++i) {
            newValues[i] = values[i] + other.values[i];
        }

        vector<size_t> newShape(shape_, shape_ + shape);
        return Tensor(newShape, newValues);
    }
    Tensor operator-(const Tensor& other) {
        if (shape != other.shape) {
            throw invalid_argument("shape must have the same number of elements");
        }

        size_t total_size = 1;
        for (size_t i = 0; i < shape; ++i) {
            total_size *= shape_[i];
        }

        vector<double> newValues(total_size);
        for (size_t i = 0; i < total_size; ++i) {
            newValues[i] = values[i] - other.values[i];
        }

        vector<size_t> newShape(shape_, shape_ + shape);
        return Tensor(newShape, newValues);
    }
    Tensor operator*(const Tensor& other) {
        if (shape != other.shape) {
            throw invalid_argument("shape must have the same number of elements");
        }

        size_t total_size = 1;
        for (size_t i = 0; i < shape; ++i) {
            total_size *= shape_[i];
        }

        vector<double> newValues(total_size);
        for (size_t i = 0; i < total_size; ++i) {
            newValues[i] = values[i] * other.values[i];
        }

        vector<size_t> newShape(shape_, shape_ + shape);
        return Tensor(newShape, newValues);
    }
    Tensor operator*(double N) {
        size_t total_size = 1;
        for (size_t i = 0; i < shape; ++i) {
            total_size *= shape_[i];
        }

        vector<double> newValues(total_size);
        for (size_t i = 0; i < total_size; ++i) {
            newValues[i] = values[i] * N;
        }

        vector<size_t> newShape(shape_, shape_ + shape);
        return Tensor(newShape, newValues);
    }

    Tensor view(const std::vector<size_t>& new_shape) {
        if (new_shape.empty() || new_shape.size() > 3) {
            throw invalid_argument("the new shape must be between 1 and 3 ");
        }

        size_t new_total = 1;
        for (size_t d : new_shape) {
            if (d == 0) throw std::invalid_argument("shape could not be 0");
            new_total *= d;
        }

        size_t tmpShape[3] = {1, 1, 1};
        for (size_t i = 0; i < new_shape.size(); ++i) tmpShape[i] = new_shape[i];

        return Tensor(new_shape.size(), tmpShape, values, ref);
    }

    Tensor unsqueeze(size_t axis) {
        if (shape >= 3) {
            throw invalid_argument("shape must be between 1 and 3 ");
        }

        if (axis > shape) {
            throw std::invalid_argument("out of range");
        }

        size_t newDim = shape + 1;

        size_t newShape[3] = {1, 1, 1};

        for (size_t i = 0; i < axis; ++i) {
            newShape[i] = shape_[i];
        }

        newShape[axis] = 1;

        for (size_t i = axis; i < shape; ++i) {
            newShape[i + 1] = shape_[i];
        }

        return Tensor(newDim, newShape, values, ref);
    }

    static Tensor concat(const vector<Tensor>& tensors, size_t axis) {

        if (tensors.empty()) {
            throw invalid_argument("empty list");
        }

        const Tensor& first = tensors[0];

        if (axis >= first.shape) {
            throw invalid_argument("axis out of range");
        }

        size_t dim = first.shape;

        for (const Tensor& t : tensors) {
            if (t.shape != dim) {
                throw invalid_argument("incompatible dimensions");
            }

            for (size_t d = 0; d < dim; ++d) {
                if (d == axis) continue;
                if (t.shape_[d] != first.shape_[d]) {
                    throw std::invalid_argument("concat: shapes incompatibles.");
                }
            }
        }


        size_t newShape[3] = {1, 1, 1};
        for (size_t d = 0; d < dim; ++d) {
            newShape[d] = first.shape_[d];
        }

        newShape[axis] = 0;
        for (const Tensor& t : tensors) {
            newShape[axis] += t.shape_[axis];
        }

        size_t total = 1;
        for (size_t d = 0; d < dim; ++d) {
            total *= newShape[d];
        }

        double* newData = new double[total];

        size_t innerStride = 1;
        for (size_t d = axis + 1; d < dim; ++d) {
            innerStride *= first.shape_[d];
        }

        size_t outerStride = 1;
        for (size_t d = 0; d < axis; ++d) {
            outerStride *= first.shape_[d];
        }

        size_t offset = 0;
        for (size_t o = 0; o < outerStride; ++o) {
            for (const Tensor& t : tensors) {
                size_t block = t.shape_[axis] * innerStride;
                size_t src = o * block;
                for (size_t i = 0; i < block; ++i) {
                    newData[offset++] = t.values[src + i];
                }
            }
        }

        Tensor result;
        result.shape = dim;
        for (size_t d = 0; d < dim; ++d) result.shape_[d] = newShape[d];
        result.values = newData;
        result.ref = new size_t(1);

        return result;
    }

    Tensor apply(const TensorTransform& transform) {
        size_t total = 1;
        for (size_t d = 0; d < shape; ++d) total *= shape_[d];

        std::vector<size_t> outShape(shape);
        for (size_t d = 0; d < shape; ++d) outShape[d] = shape_[d];

        std::vector<double> outValues(total);
        for (size_t i = 0; i < total; ++i) {
            outValues[i] = transform.apply(values[i]);
        }

        return Tensor(outShape, outValues);
    }

    vector<size_t> getShape() const {
        return vector<size_t>(shape_, shape_ + shape);
    }

    void print_shape(const string& label = "") const {
        if (!label.empty()) cout << label << " ";
        cout << "{";
        for (size_t i = 0; i < shape; ++i) {
            cout << shape_[i];
            if (i + 1 < shape) std::cout << ", ";
        }
        cout << "}\n";
    }

    Tensor add_bias_row(const Tensor& bias) {
        size_t M = shape_[0];
        size_t N = shape_[1];

        vector<size_t> outShape = {M, N};
        vector<double> outValues(M * N);

        for (size_t i = 0; i < M; ++i) {
            for (size_t j = 0; j < N; ++j) {
                size_t idx = i * N + j;
                outValues[idx] = values[idx] + bias.values[j]; // bias fila 0
            }
        }

        return Tensor(outShape, outValues);
    }
};

Tensor dot(const Tensor& a, const Tensor& b)
{
    if (a.shape != b.shape) {
        throw invalid_argument("incompatible dimensions");
    }
    for (size_t d = 0; d < a.shape; ++d) {
        if (a.shape_[d] != b.shape_[d]) {
            throw invalid_argument("incompatible shapes");
        }
    }

    size_t total = 1;
    for (size_t d = 0; d < a.shape; ++d) total *= a.shape_[d];

    double acc = 0.0;
    for (size_t i = 0; i < total; ++i) {
        acc += a.values[i] * b.values[i];
    }

    std::vector<size_t> outShape = {1};
    std::vector<double> outValues = {acc};
    return Tensor(outShape, outValues);
}

Tensor matmul(const Tensor& a, const Tensor& b)
{
    if (a.shape != 2 || b.shape != 2) {
        throw std::invalid_argument("both tensors must be 2D");
    }

    size_t m = a.shape_[0];
    size_t n = a.shape_[1];
    size_t n2 = b.shape_[0];
    size_t p = b.shape_[1];

    if (n != n2) {
        throw std::invalid_argument("incompatible shapes");
    }

    std::vector<size_t> outShape = {m, p};
    std::vector<double> outValues(m * p, 0.0);

    for (size_t i = 0; i < m; ++i) {
        for (size_t k = 0; k < n; ++k) {
            double aik = a.values[i * n + k];
            for (size_t j = 0; j < p; ++j) {
                outValues[i * p + j] += aik * b.values[k * p + j];
            }
        }
    }

    return Tensor(outShape, outValues);
}

int main() {

    Tensor X = Tensor::random({1000, 20, 20}, 0.0, 1.0);
    X.print_shape("Paso 1:");

    Tensor Xflat = X.view({1000, 400});
    Xflat.print_shape("Paso 2:");

    Tensor W1 = Tensor::random({400, 100}, -0.1, 0.1);
    Tensor Z1 = matmul(Xflat, W1);
    Z1.print_shape("Paso 3:");

    Tensor b1 = Tensor::random({1, 100}, -0.1, 0.1);
    Tensor Z1b = Z1.add_bias_row(b1);
    Z1b.print_shape("Paso 4:");

    ReLU relu;
    Tensor A1 = Z1b.apply(relu);
    A1.print_shape("Paso 5:");

    Tensor W2 = Tensor::random({100, 10}, -0.1, 0.1);
    Tensor Z2 = matmul(A1, W2);
    Z2.print_shape("Paso 6:");

    Tensor b2 = Tensor::random({1, 10}, -0.1, 0.1);
    Tensor Z2b = Z2.add_bias_row(b2);
    Z2b.print_shape("Paso 7:");

    Sigmoid sigmoid;
    Tensor Y = Z2b.apply(sigmoid);
    Y.print_shape("Paso 8:");

    return 0;
}