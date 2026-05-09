#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <memory>
#include <string>
#include <unordered_map>
#include <string_view>

#include "re2/re2.h"
#include "re2/stringpiece.h"

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

struct DecodedCodepoint {
    Py_UCS4 value;
    std::size_t byte_length;
};

static DecodedCodepoint decode_utf8(std::string_view text, std::size_t pos) {
    const unsigned char lead = static_cast<unsigned char>(text[pos]);
    if ((lead & 0x80) == 0) {
        return {lead, 1};
    }
    if ((lead & 0xE0) == 0xC0) {
        return {
            static_cast<Py_UCS4>(((lead & 0x1F) << 6) |
            (static_cast<unsigned char>(text[pos + 1]) & 0x3F)),
            2
        };
    }
    if ((lead & 0xF0) == 0xE0) {
        return {
            static_cast<Py_UCS4>(((lead & 0x0F) << 12) |
            ((static_cast<unsigned char>(text[pos + 1]) & 0x3F) << 6) |
            (static_cast<unsigned char>(text[pos + 2]) & 0x3F)),
            3
        };
    }
    return {
        static_cast<Py_UCS4>(((lead & 0x07) << 18) |
        ((static_cast<unsigned char>(text[pos + 1]) & 0x3F) << 12) |
        ((static_cast<unsigned char>(text[pos + 2]) & 0x3F) << 6) |
        (static_cast<unsigned char>(text[pos + 3]) & 0x3F)),
        4
    };
}

static bool starts_with_contraction(std::string_view text, std::size_t pos, std::size_t* length) {
    if (text[pos] != '\'') {
        return false;
    }
    const std::string_view tail = text.substr(pos);
    if (tail.substr(0, 3) == "'ll" || tail.substr(0, 3) == "'ve" || tail.substr(0, 3) == "'re") {
        *length = 3;
        return true;
    }
    if (tail.size() >= 2) {
        const char suffix = tail[1];
        if (suffix == 's' || suffix == 'd' || suffix == 'm' || suffix == 't') {
            *length = 2;
            return true;
        }
    }
    return false;
}

static bool is_whitespace(Py_UCS4 value) {
    return Py_UNICODE_ISSPACE(value);
}

static bool is_letter(Py_UCS4 value) {
    return Py_UNICODE_ISALPHA(value);
}

static bool is_number(Py_UCS4 value) {
    return Py_UNICODE_ISNUMERIC(value);
}

static void add_token(
    std::unordered_map<std::string, Py_ssize_t>& counts,
    std::string_view text,
    std::size_t start,
    std::size_t end
) {
    counts[std::string(text.substr(start, end - start))] += 1;
}

static void tokenize_doc_into_map(std::string_view text, std::unordered_map<std::string, Py_ssize_t>& counts) {
    std::size_t pos = 0;
    while (pos < text.size()) {
        std::size_t contraction_length = 0;
        if (starts_with_contraction(text, pos, &contraction_length)) {
            add_token(counts, text, pos, pos + contraction_length);
            pos += contraction_length;
            continue;
        }

        const DecodedCodepoint current = decode_utf8(text, pos);
        const std::size_t next_pos = pos + current.byte_length;

        if (current.value == ' ' && next_pos < text.size()) {
            const DecodedCodepoint next = decode_utf8(text, next_pos);
            if (is_letter(next.value)) {
                std::size_t end = next_pos + next.byte_length;
                while (end < text.size()) {
                    const DecodedCodepoint cp = decode_utf8(text, end);
                    if (!is_letter(cp.value)) {
                        break;
                    }
                    end += cp.byte_length;
                }
                add_token(counts, text, pos, end);
                pos = end;
                continue;
            }
            if (is_number(next.value)) {
                std::size_t end = next_pos + next.byte_length;
                while (end < text.size()) {
                    const DecodedCodepoint cp = decode_utf8(text, end);
                    if (!is_number(cp.value)) {
                        break;
                    }
                    end += cp.byte_length;
                }
                add_token(counts, text, pos, end);
                pos = end;
                continue;
            }
            if (!is_whitespace(next.value) && !is_letter(next.value) && !is_number(next.value)) {
                std::size_t end = next_pos + next.byte_length;
                while (end < text.size()) {
                    const DecodedCodepoint cp = decode_utf8(text, end);
                    if (is_whitespace(cp.value) || is_letter(cp.value) || is_number(cp.value)) {
                        break;
                    }
                    end += cp.byte_length;
                }
                add_token(counts, text, pos, end);
                pos = end;
                continue;
            }
        }

        if (is_letter(current.value)) {
            std::size_t end = next_pos;
            while (end < text.size()) {
                const DecodedCodepoint cp = decode_utf8(text, end);
                if (!is_letter(cp.value)) {
                    break;
                }
                end += cp.byte_length;
            }
            add_token(counts, text, pos, end);
            pos = end;
            continue;
        }

        if (is_number(current.value)) {
            std::size_t end = next_pos;
            while (end < text.size()) {
                const DecodedCodepoint cp = decode_utf8(text, end);
                if (!is_number(cp.value)) {
                    break;
                }
                end += cp.byte_length;
            }
            add_token(counts, text, pos, end);
            pos = end;
            continue;
        }

        if (is_whitespace(current.value)) {
            std::size_t end = next_pos;
            std::size_t last_start = pos;
            while (end < text.size()) {
                const DecodedCodepoint cp = decode_utf8(text, end);
                if (!is_whitespace(cp.value)) {
                    break;
                }
                last_start = end;
                end += cp.byte_length;
            }
            if (end == text.size() || last_start == pos) {
                add_token(counts, text, pos, end);
                pos = end;
            } else {
                add_token(counts, text, pos, last_start);
                pos = last_start;
            }
            continue;
        }

        std::size_t end = next_pos;
        while (end < text.size()) {
            const DecodedCodepoint cp = decode_utf8(text, end);
            if (is_whitespace(cp.value) || is_letter(cp.value) || is_number(cp.value)) {
                break;
            }
            end += cp.byte_length;
        }
        add_token(counts, text, pos, end);
        pos = end;
    }
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
    re2::StringPiece full_text(text_string);
    re2::StringPiece submatches[2];
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
        if (match_end <= search_start) {
            search_start += 1;
        } else {
            search_start = match_end;
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

    (void)pattern;

    std::unordered_map<std::string, Py_ssize_t> counts;
    std::string_view raw_data(data, static_cast<size_t>(data_len));
    std::string_view raw_split(split_token, static_cast<size_t>(split_token_len));
    size_t start = 0;

    while (start <= raw_data.size()) {
        size_t next = raw_data.find(raw_split, start);
        std::string_view raw_doc = next == std::string_view::npos
            ? raw_data.substr(start)
            : raw_data.substr(start, next - start);

        PyObject* decoded = PyUnicode_DecodeUTF8(
            raw_doc.data(),
            static_cast<Py_ssize_t>(raw_doc.size()),
            "ignore"
        );
        if (decoded == nullptr) {
            return nullptr;
        }

        Py_ssize_t cleaned_len = 0;
        const char* cleaned_text = PyUnicode_AsUTF8AndSize(decoded, &cleaned_len);
        if (cleaned_text == nullptr) {
            Py_DECREF(decoded);
            return nullptr;
        }

        tokenize_doc_into_map(std::string_view(cleaned_text, static_cast<size_t>(cleaned_len)), counts);
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
