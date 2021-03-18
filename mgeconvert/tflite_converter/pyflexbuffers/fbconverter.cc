#include <pybind11/pybind11.h>
#include <flatbuffers/flexbuffers.h>
#include <flatbuffers/idl.h>
#include <string>

std::string ToJSON(pybind11::bytes flexbuf) {
    auto flexbuf_str = std::string(flexbuf);
    auto fb = flexbuffers::GetRoot(
            reinterpret_cast<const uint8_t *>(flexbuf_str.data()),
            flexbuf_str.size());
    
    std::string result;
    fb.ToString(true, true, result);
    return result;
}

pybind11::bytes FromJSON(const std::string& json_str) {
    flatbuffers::Parser parser;
    flexbuffers::Builder builder;
    if (!parser.ParseFlexBuffer(json_str.data(), nullptr, &builder)) {
        throw std::invalid_argument("Failed to parse FlexBuffers.");
    }
    
    auto flexbuf = builder.GetBuffer();
    return {reinterpret_cast<const char *>(flexbuf.data()), flexbuf.size()};
}

PYBIND11_MODULE(fbconverter, m) {
    m.def("to_json", &ToJSON,
            "Converts FlexBuffers to JSON. Takes bytes, returns str.");
    m.def("from_json", &FromJSON,
            "Converts JSON to FlexBuffers. Takes str, returns bytes.");
}
