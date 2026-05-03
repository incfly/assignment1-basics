#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <algorithm>
#include <memory>
#include <string>
#include <unordered_map>
#include <string_view>

#include "re2/re2.h"

static std::unordered_map<std::string, std::unique_ptr<re2::RE2>> g_re_cache;

static re2::RE2* get_compiled_re(const std::string& pattern) {
    auto it = g_re_cache.find(pattern);
    if (it != g_re_cache.end()) {
        return it->second.get();
    }

    auto compiled = std::make_unique<re2::RE2>(pattern);
    re2::RE2* compiled_ptr = compiled.get();
    g_re_cache.emplace(pattern, std::move(compiled));
    return compiled_ptr;
}

static bool count_matches_into_map(
    re2::RE2* re,
    absl::string_view text,
    std::unordered_map<std::string, Py_ssize_t>& counts
) {
    absl::string_view submatches[2];
    size_t search_start = 0;

    while (search_start <= text.size() &&
           re->Match(text, search_start, text.size(), re2::RE2::UNANCHORED, submatches, 2)) {
        counts[std::string(submatches[1])] += 1;

        size_t match_begin = static_cast<size_t>(submatches[0].data() - text.data());
        size_t match_end = match_begin + submatches[0].size();
        search_start = match_end;
        if (match_end == match_begin) {
            search_start += 1;
        }
    }

    return true;
}

static PyObject* findall(PyObject* self, PyObject* args) {
    const char* pattern = nullptr;
    const char* text = nullptr;

    if (!PyArg_ParseTuple(args, "ss", &pattern, &text)) {
        return nullptr;
    }

    std::string wrapped_pattern = "(" + std::string(pattern) + ")";
    re2::RE2* re = get_compiled_re(wrapped_pattern);
    if (!re->ok()) {
        PyErr_Format(PyExc_ValueError, "invalid RE2 pattern: %s", re->error().c_str());
        return nullptr;
    }

    PyObject* out = PyList_New(0);
    if (out == nullptr) {
        return nullptr;
    }

    std::string text_string(text);
    absl::string_view full_text(text_string);
    absl::string_view submatches[2];
    size_t search_start = 0;

    while (search_start <= full_text.size() &&
           re->Match(full_text, search_start, full_text.size(), re2::RE2::UNANCHORED, submatches, 2)) {
        PyObject* py_match = PyUnicode_FromStringAndSize(
            submatches[1].data(),
            static_cast<Py_ssize_t>(submatches[1].size())
        );
        if (py_match == nullptr) {
            Py_DECREF(out);
            return nullptr;
        }
        if (PyList_Append(out, py_match) < 0) {
            Py_DECREF(py_match);
            Py_DECREF(out);
            return nullptr;
        }
        Py_DECREF(py_match);

        size_t match_begin = static_cast<size_t>(submatches[0].data() - full_text.data());
        size_t match_end = match_begin + submatches[0].size();
        search_start = match_end;
        if (match_end == match_begin) {
            search_start += 1;
        }
    }

    return out;
}

static PyObject* token_freqmap(PyObject* self, PyObject* args) {
    const char* pattern = nullptr;
    const char* data = nullptr;
    const char* split_token = nullptr;
    Py_ssize_t data_len = 0;
    Py_ssize_t split_token_len = 0;

    if (!PyArg_ParseTuple(args, "sy#y#", &pattern, &data, &data_len, &split_token, &split_token_len)) {
        return nullptr;
    }
    if (split_token_len <= 0) {
        PyErr_SetString(PyExc_ValueError, "split_token must not be empty");
        return nullptr;
    }

    std::string wrapped_pattern = "(" + std::string(pattern) + ")";
    re2::RE2* re = get_compiled_re(wrapped_pattern);
    if (!re->ok()) {
        PyErr_Format(PyExc_ValueError, "invalid RE2 pattern: %s", re->error().c_str());
        return nullptr;
    }

    std::unordered_map<std::string, Py_ssize_t> counts;
    std::string_view raw_data(data, static_cast<size_t>(data_len));
    std::string_view raw_split(split_token, static_cast<size_t>(split_token_len));
    size_t start = 0;

    while (start <= raw_data.size()) {
        size_t next = raw_data.find(raw_split, start);
        std::string_view raw_doc = next == std::string_view::npos
            ? raw_data.substr(start)
            : raw_data.substr(start, next - start);

        PyObject* decoded = PyUnicode_DecodeUTF8(raw_doc.data(), static_cast<Py_ssize_t>(raw_doc.size()), "ignore");
        if (decoded == nullptr) {
            return nullptr;
        }

        Py_ssize_t cleaned_len = 0;
        const char* cleaned_text = PyUnicode_AsUTF8AndSize(decoded, &cleaned_len);
        if (cleaned_text == nullptr) {
            Py_DECREF(decoded);
            return nullptr;
        }

        count_matches_into_map(
            re,
            absl::string_view(cleaned_text, static_cast<size_t>(cleaned_len)),
            counts
        );
        Py_DECREF(decoded);

        if (next == std::string_view::npos) {
            break;
        }
        start = next + raw_split.size();
    }

    PyObject* out = PyDict_New();
    if (out == nullptr) {
        return nullptr;
    }

    for (const auto& [token, count] : counts) {
        PyObject* key = PyBytes_FromStringAndSize(token.data(), static_cast<Py_ssize_t>(token.size()));
        if (key == nullptr) {
            Py_DECREF(out);
            return nullptr;
        }
        PyObject* value = PyLong_FromSsize_t(count);
        if (value == nullptr) {
            Py_DECREF(key);
            Py_DECREF(out);
            return nullptr;
        }
        if (PyDict_SetItem(out, key, value) < 0) {
            Py_DECREF(key);
            Py_DECREF(value);
            Py_DECREF(out);
            return nullptr;
        }
        Py_DECREF(key);
        Py_DECREF(value);
    }

    return out;
}

static PyMethodDef re2_demo_methods[] = {
    {"findall", findall, METH_VARARGS, "Return all non-overlapping full matches using RE2."},
    {"token_freqmap", token_freqmap, METH_VARARGS, "Return a token frequency map for a whole chunk using RE2."},
    {nullptr, nullptr, 0, nullptr},
};

static PyModuleDef re2_demo_module = {
    PyModuleDef_HEAD_INIT,
    "_re2demo",
    "Minimal Python binding for Google RE2.",
    -1,
    re2_demo_methods,
};

PyMODINIT_FUNC PyInit__re2demo(void) {
    return PyModule_Create(&re2_demo_module);
}
