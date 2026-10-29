#include <pybind11/pybind11.h>
#include <pybind11/stl.h> 
#include <string_view>
#include <vector>
#include <algorithm>
#include <cctype>

namespace py = pybind11;

// Returns a vector of {start_index, length} to avoid intermediate string allocations
std::vector<std::pair<size_t, size_t>> get_chunk_boundaries(std::string_view text, size_t chunk_size, size_t overlap) {
    std::vector<std::pair<size_t, size_t>> boundaries;
    if (text.empty() || chunk_size == 0) return boundaries;

    size_t start = 0;
    size_t text_length = text.length();

    size_t step = (chunk_size > overlap) ? (chunk_size - overlap) : 1;
    boundaries.reserve((text_length / step) + 1);

    while (start < text_length) {
        size_t end = std::min(start + chunk_size, text_length);
        
        // Trim leading whitespace
        size_t chunk_start = start;
        while (chunk_start < end && std::isspace(static_cast<unsigned char>(text[chunk_start]))) {
            chunk_start++;
        }
        
        // Trim trailing whitespace
        size_t chunk_end = end;
        while (chunk_end > chunk_start && std::isspace(static_cast<unsigned char>(text[chunk_end - 1]))) {
            chunk_end--;
        }

        if (chunk_start < chunk_end) {
            // Store the offset and length without allocating a string
            boundaries.emplace_back(chunk_start, chunk_end - chunk_start);
        }

        if (chunk_size <= overlap) break; 
        start += step;
    }

    return boundaries;
}

PYBIND11_MODULE(localmind_cpp, m) {
    m.doc() = "C++ optimized backend for LocalMind";
    
    m.def("get_chunk_boundaries", &get_chunk_boundaries, 
          "Calculate zero-copy sliding window boundaries.",
          py::arg("text"), py::arg("chunk_size"), py::arg("overlap"));
}