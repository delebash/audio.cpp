// Added in delebash/audio.cpp (JustVoice's copy of audio.cpp), 2026-10-03: MeCab over a UniDic
// dictionary, shared by Kokoro's and Chatterbox's Japanese. The library loading and the node ABI
// prefix follow src/models/kokoro_tts/g2p_multilingual.cpp. 2026-10-04: macOS and Linux load the
// libmecab beside the executable first, as Windows does, then the system's.
#include "engine/framework/text/mecab.h"

#include <cstdint>
#include <cstdlib>
#include <mutex>
#include <stdexcept>

#if defined(__APPLE__)
#include <mach-o/dyld.h>
#elif !defined(_WIN32)
#include <unistd.h>
#endif
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace engine::text {

namespace {

bool has_dicrc(const std::filesystem::path & dir) {
    std::error_code ec;
    return !dir.empty() && std::filesystem::is_regular_file(dir / "dicrc", ec);
}

class Library {
public:
    Library() {
        const char * override_path = std::getenv("AUDIOCPP_MECAB_LIBRARY");
#ifdef _WIN32
        std::filesystem::path path;
        if (override_path && *override_path) {
            path = std::filesystem::u8path(override_path);
        } else {
            std::wstring exe(32768, L'\0');
            const auto n = GetModuleFileNameW(nullptr, exe.data(), static_cast<DWORD>(exe.size()));
            if (!n || n >= exe.size()) throw std::runtime_error("Cannot locate the audio.cpp executable");
            exe.resize(n);
            path = std::filesystem::path(exe).parent_path() / "libmecab.dll";
        }
        handle_ = LoadLibraryExW(path.c_str(), nullptr, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
#else
        if (override_path && *override_path) {
            handle_ = dlopen(override_path, RTLD_NOW | RTLD_LOCAL);
        } else {
            // The release stages libmecab beside the executable on every platform
            // (cmake/text_dictionaries.cmake); a bare name only searches the system paths.
#if defined(__APPLE__)
            const char * name = "libmecab.2.dylib";
#else
            const char * name = "libmecab.so.2";
#endif
            std::error_code ec;
            const auto beside = executable_directory() / name;
            if (std::filesystem::is_regular_file(beside, ec)) {
                handle_ = dlopen(beside.c_str(), RTLD_NOW | RTLD_LOCAL);
            }
            if (!handle_) handle_ = dlopen(name, RTLD_NOW | RTLD_LOCAL);
        }
#endif
        if (!handle_) {
            throw std::runtime_error("Japanese needs libmecab beside the audio.cpp executable (or set AUDIOCPP_MECAB_LIBRARY)");
        }
    }
    ~Library() {
#ifdef _WIN32
        if (handle_) FreeLibrary(handle_);
#else
        if (handle_) dlclose(handle_);
#endif
    }
    template <class F> F symbol(const char * name) const {
#ifdef _WIN32
        auto * p = reinterpret_cast<void *>(GetProcAddress(handle_, name));
#else
        auto * p = dlsym(handle_, name);
#endif
        if (!p) throw std::runtime_error(std::string("libmecab lacks ") + name);
        return reinterpret_cast<F>(p);
    }

private:
#ifdef _WIN32
    HMODULE handle_ = nullptr;
#else
    void * handle_ = nullptr;
#endif
};

// Prefix of MeCab's stable C node ABI; later fields are not accessed.
struct Node {
    Node * prev; Node * next; Node * enext; Node * bnext;
    void * rpath; void * lpath;
    const char * surface; const char * feature;
    unsigned int id;
    unsigned short length, rlength, rcAttr, lcAttr, posid;
    unsigned char char_type, stat;
};

std::vector<std::string> split_commas(const char * text) {
    std::vector<std::string> out;
    std::string current;
    for (const char * p = text; p && *p; ++p) {
        if (*p == ',') {
            out.push_back(current);
            current.clear();
        } else {
            current.push_back(*p);
        }
    }
    out.push_back(current);
    return out;
}

}  // namespace

std::filesystem::path executable_directory() {
    // As src/framework/text/espeak_phonemizer.cpp finds eSpeak's data.
#ifdef _WIN32
    std::vector<wchar_t> buffer(32768);
    const auto length = GetModuleFileNameW(nullptr, buffer.data(), static_cast<DWORD>(buffer.size()));
    if (length && length < buffer.size()) return std::filesystem::path(std::wstring(buffer.data(), length)).parent_path();
#elif defined(__APPLE__)
    uint32_t size = 0;
    _NSGetExecutablePath(nullptr, &size);
    std::vector<char> buffer(size);
    if (_NSGetExecutablePath(buffer.data(), &size) == 0)
        return std::filesystem::weakly_canonical(buffer.data()).parent_path();
#else
    std::vector<char> buffer(4096);
    for (;;) {
        const auto size = readlink("/proc/self/exe", buffer.data(), buffer.size());
        if (size < 0) break;
        if (static_cast<size_t>(size) < buffer.size())
            return std::filesystem::path(std::string(buffer.data(), size)).parent_path();
        buffer.resize(buffer.size() * 2);
    }
#endif
    throw std::runtime_error("Cannot locate the audio.cpp executable");
}

std::optional<std::filesystem::path> unidic_dictionary_dir(const std::filesystem::path & embedded) {
    if (has_dicrc(embedded)) return embedded;
    const char * env = std::getenv("AUDIOCPP_UNIDIC_DIR");
    if (env && *env) {
        const auto dir = std::filesystem::u8path(env);
        if (has_dicrc(dir)) return dir;
    }
    return std::nullopt;
}

struct MecabTagger::Impl {
    Library lib;
    void * tagger = nullptr;
    void (*destroy)(void *) = nullptr;
    const Node * (*parse)(void *, const char *) = nullptr;
    std::mutex mutex;
};

MecabTagger::MecabTagger(const std::filesystem::path & unidic_dir) : impl_(std::make_unique<Impl>()) {
    auto create = impl_->lib.symbol<void * (*)(int, char **)>("mecab_new");
    auto error = impl_->lib.symbol<const char * (*)(void *)>("mecab_strerror");
    impl_->destroy = impl_->lib.symbol<void (*)(void *)>("mecab_destroy");
    impl_->parse = impl_->lib.symbol<const Node * (*)(void *, const char *)>("mecab_sparse_tonode");
    std::vector<std::string> args = {"mecab", "-r", (unidic_dir / "dicrc").u8string(), "-d", unidic_dir.u8string()};
    std::vector<char *> argv;
    for (auto & arg : args) argv.push_back(arg.data());
    impl_->tagger = create(static_cast<int>(argv.size()), argv.data());
    if (!impl_->tagger) {
        throw std::runtime_error(std::string("Cannot load the Japanese dictionary: ") + error(nullptr));
    }
}

MecabTagger::~MecabTagger() {
    if (impl_ && impl_->tagger) impl_->destroy(impl_->tagger);
}

std::vector<MecabToken> MecabTagger::parse(const std::string & text) const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    const Node * result = impl_->parse(impl_->tagger, text.c_str());
    if (!result) throw std::runtime_error("Japanese tokenization failed");
    std::vector<MecabToken> out;
    for (const Node * node = result; node; node = node->next) {
        if (node->stat >= 2) continue;  // BOS / EOS
        MecabToken token;
        token.surface.assign(node->surface, node->length);
        token.features = split_commas(node->feature);
        token.begin = static_cast<size_t>(node->surface - text.c_str());
        token.end = token.begin + node->length;
        out.push_back(std::move(token));
    }
    return out;
}

}  // namespace engine::text
