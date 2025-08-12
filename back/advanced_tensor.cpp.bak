#include "advanced_gpt.hpp"
#include <stdexcept>
#include <algorithm>

namespace gpt {

// AdvancedTensor 메서드 구현
AdvancedTensor AdvancedTensor::matmul(const AdvancedTensor& other) const {
    if (shape_.size() != 2 || other.shape_.size() != 2) {
        throw std::runtime_error("matmul only supports 2D tensors");
    }
    
    int m = shape_[0];
    int k = shape_[1];
    int n = other.shape_[1];
    
    if (k != other.shape_[0]) {
        throw std::runtime_error("matrix dimensions do not match for multiplication");
    }
    
    AdvancedTensor result({m, n}, device_, requires_grad_);
    
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (int l = 0; l < k; ++l) {
                sum += data_[i * k + l] * other.data_[l * n + j];
            }
            result.data_[i * n + j] = sum;
        }
    }
    
    return result;
}

AdvancedTensor AdvancedTensor::transpose(int dim0, int dim1) const {
    if (shape_.size() != 2) {
        throw std::runtime_error("transpose only supports 2D tensors for now");
    }
    
    if (dim0 != 0 || dim1 != 1) {
        throw std::runtime_error("only (0,1) transpose supported");
    }
    
    AdvancedTensor result({shape_[1], shape_[0]}, device_, requires_grad_);
    
    for (int i = 0; i < shape_[0]; ++i) {
        for (int j = 0; j < shape_[1]; ++j) {
            result.data_[j * shape_[0] + i] = data_[i * shape_[1] + j];
        }
    }
    
    return result;
}

AdvancedTensor AdvancedTensor::view(const std::vector<int>& new_shape) const {
    int new_size = 1;
    for (int dim : new_shape) {
        new_size *= dim;
    }
    
    if (new_size != numel()) {
        throw std::runtime_error("view size does not match tensor size");
    }
    
    AdvancedTensor result = *this;
    result.shape_ = new_shape;
    return result;
}

void AdvancedTensor::add_(const AdvancedTensor& other) {
    if (shape_ != other.shape_) {
        throw std::runtime_error("tensor shapes do not match for addition");
    }
    
    for (size_t i = 0; i < data_.size(); ++i) {
        data_[i] += other.data_[i];
    }
}

void AdvancedTensor::mul_(float scalar) {
    for (auto& val : data_) {
        val *= scalar;
    }
}

void AdvancedTensor::div_(float scalar) {
    if (scalar == 0.0f) {
        throw std::runtime_error("division by zero");
    }
    
    for (auto& val : data_) {
        val /= scalar;
    }
}

} // namespace gpt
