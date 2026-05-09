// This program demosntrate that with and without capture group, the tokenizing perf can
// matters up to 17x times! with capturing still doing one match at a time, but some re2
// internals make the perf number varies a lot.
#include <chrono>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>

#include "re2/re2.h"
#include "re2/stringpiece.h"

namespace {

constexpr std::string_view kSplitToken = "<|endoftext|>";
constexpr std::string_view kPattern =
    R"('(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+)";

struct Result {
    double seconds = 0.0;
    std::uint64_t matches = 0;
    std::uint64_t token_bytes = 0;
};

std::string read_file(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("failed to open corpus: " + path);
    }
    file.seekg(0, std::ios::end);
    const auto size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::string data(static_cast<std::size_t>(size), '\0');
    if (!file.read(data.data(), size)) {
        throw std::runtime_error("failed to read corpus: " + path);
    }
    return data;
}

Result run_scan(re2::RE2* re, std::string_view corpus, bool capture_group) {
    Result result;
    const auto start_time = std::chrono::steady_clock::now();

    std::size_t doc_start = 0;
    while (doc_start <= corpus.size()) {
        const std::size_t doc_end = corpus.find(kSplitToken, doc_start);
        const std::string_view doc = doc_end == std::string_view::npos
            ? corpus.substr(doc_start)
            : corpus.substr(doc_start, doc_end - doc_start);

        re2::StringPiece full_text(doc.data(), doc.size());
        re2::StringPiece submatches[2];
        const int nsubmatch = capture_group ? 2 : 1;
        std::size_t search_start = 0;

        while (search_start <= full_text.size() &&
               re->Match(full_text, search_start, full_text.size(), re2::RE2::UNANCHORED, submatches, nsubmatch)) {
            const re2::StringPiece token = capture_group ? submatches[1] : submatches[0];
            result.matches += 1;
            result.token_bytes += token.size();

            const std::size_t match_begin =
                static_cast<std::size_t>(submatches[0].data() - full_text.data());
            const std::size_t match_end = match_begin + submatches[0].size();
            if (match_end <= search_start) {
                search_start += 1;
            } else {
                search_start = match_end;
            }
        }

        if (doc_end == std::string_view::npos) {
            break;
        }
        doc_start = doc_end + kSplitToken.size();
    }

    result.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start_time).count();
    return result;
}

void print_result(const std::string& label, const Result& result) {
    std::cout << label
              << " matches=" << result.matches
              << " token_bytes=" << result.token_bytes
              << " seconds=" << result.seconds
              << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: " << argv[0] << " <corpus>\n";
        return 1;
    }

    try {
        const std::string corpus = read_file(argv[1]);
        re2::RE2 capture_re{"(" + std::string(kPattern) + ")"};
        re2::RE2 nocapture_re{std::string(kPattern)};

        if (!capture_re.ok()) {
            std::cerr << "capture regex error: " << capture_re.error() << '\n';
            return 1;
        }
        if (!nocapture_re.ok()) {
            std::cerr << "no-capture regex error: " << nocapture_re.error() << '\n';
            return 1;
        }

        std::cout << "corpus_bytes=" << corpus.size() << '\n';
        print_result("capture_group", run_scan(&capture_re, corpus, true));
        print_result("no_capture", run_scan(&nocapture_re, corpus, false));
    } catch (const std::exception& exc) {
        std::cerr << exc.what() << '\n';
        return 1;
    }
    return 0;
}
