// GKDT ggml runtime - SimpleTokenizer (CLIP-style BPE) for dinotxt.
#include "tokenizer.hpp"
#include "common.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>

namespace gkd {

// ---------------------------------------------------------------------------
// UTF-8 helpers
// ---------------------------------------------------------------------------
namespace {

// Decode one UTF-8 code point starting at s[i]; advances i.
static uint32_t utf8_next(const std::string& s, size_t& i) {
    unsigned char c = (unsigned char)s[i];
    if (c < 0x80) {
        i += 1;
        return c;
    }
    int len = 0;
    uint32_t cp = 0;
    if ((c & 0xE0) == 0xC0) { len = 2; cp = c & 0x1F; }
    else if ((c & 0xF0) == 0xE0) { len = 3; cp = c & 0x0F; }
    else if ((c & 0xF8) == 0xF0) { len = 4; cp = c & 0x07; }
    else { i += 1; return 0xFFFD; }
    if (i + len > s.size()) { i = s.size(); return 0xFFFD; }
    for (int k = 1; k < len; k++) {
        unsigned char cc = (unsigned char)s[i + k];
        if ((cc & 0xC0) != 0x80) { i += 1; return 0xFFFD; }
        cp = (cp << 6) | (cc & 0x3F);
    }
    i += len;
    return cp;
}

static void utf8_append(std::string& s, uint32_t cp) {
    if (cp < 0x80) {
        s += (char)cp;
    } else if (cp < 0x800) {
        s += (char)(0xC0 | (cp >> 6));
        s += (char)(0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
        s += (char)(0xE0 | (cp >> 12));
        s += (char)(0x80 | ((cp >> 6) & 0x3F));
        s += (char)(0x80 | (cp & 0x3F));
    } else {
        s += (char)(0xF0 | (cp >> 18));
        s += (char)(0x80 | ((cp >> 12) & 0x3F));
        s += (char)(0x80 | ((cp >> 6) & 0x3F));
        s += (char)(0x80 | (cp & 0x3F));
    }
}

// Unicode category approximation used by the CLIP regex
//   <\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+
// Covers ASCII exactly; non-ASCII code points are classified with the common
// letter/digit/space ranges and default to letter (documented approximation).
enum class Cat { Space, Letter, Digit, Other };

static Cat category(uint32_t cp) {
    if (cp == ' ' || (cp >= 0x09 && cp <= 0x0D)) return Cat::Space;
    if (cp < 0x80) {
        if ((cp >= 'a' && cp <= 'z') || (cp >= 'A' && cp <= 'Z')) return Cat::Letter;
        if (cp >= '0' && cp <= '9') return Cat::Digit;
        return Cat::Other;
    }
    // unicode whitespace
    if (cp == 0xA0 || cp == 0x1680 || (cp >= 0x2000 && cp <= 0x200A) ||
        cp == 0x2028 || cp == 0x2029 || cp == 0x202F || cp == 0x205F || cp == 0x3000)
        return Cat::Space;
    // digits (Nd)
    if ((cp >= 0x0660 && cp <= 0x0669) || (cp >= 0x06F0 && cp <= 0x06F9) ||
        (cp >= 0x0966 && cp <= 0x096F) || (cp >= 0xFF10 && cp <= 0xFF19))
        return Cat::Digit;
    // letters: latin-1 supplement/extended, greek, cyrillic, hebrew, arabic,
    // devanagari, CJK, kana, hangul and general fallback
    if ((cp >= 0xC0 && cp <= 0x24F && cp != 0xD7 && cp != 0xF7) ||
        (cp >= 0x370 && cp <= 0x3FF) || (cp >= 0x400 && cp <= 0x4FF) ||
        (cp >= 0x531 && cp <= 0x58F) || (cp >= 0x590 && cp <= 0x6FF) ||
        (cp >= 0x900 && cp <= 0x97F) || (cp >= 0x1E00 && cp <= 0x1FFF) ||
        (cp >= 0x2C60 && cp <= 0x2C7F) || (cp >= 0x3040 && cp <= 0x30FF) ||
        (cp >= 0x3400 && cp <= 0x4DBF) || (cp >= 0x4E00 && cp <= 0x9FFF) ||
        (cp >= 0xAC00 && cp <= 0xD7AF) || (cp >= 0xF900 && cp <= 0xFAFF) ||
        (cp >= 0xFF21 && cp <= 0xFF3A) || (cp >= 0xFF41 && cp <= 0xFF5A) ||
        (cp >= 0xFF66 && cp <= 0xFF9D))
        return Cat::Letter;
    return Cat::Letter;  // default: treat unknown as letter
}

// minimal html.unescape for the entities that matter in practice
static void html_unescape(std::string& s) {
    const std::pair<const char*, const char*> ents[] = {
        {"&amp;", "&"}, {"&lt;", "<"}, {"&gt;", ">"}, {"&quot;", "\""},
        {"&apos;", "'"}, {"&nbsp;", " "}, {"&#39;", "'"}, {"&#38;", "&"},
        {"&ndash;", "-"}, {"&mdash;", "-"},
    };
    for (auto& [from, to] : ents) {
        size_t pos;
        while ((pos = s.find(from)) != std::string::npos) {
            s.replace(pos, std::strlen(from), to);
        }
    }
}

// lowercase (ASCII exact; full-width/simple unicode ranges folded)
static void lower_ascii(std::string& s) {
    for (auto& c : s) {
        if (c >= 'A' && c <= 'Z') c = c - 'A' + 'a';
    }
}

}  // namespace

// ---------------------------------------------------------------------------
// SimpleTokenizer
// ---------------------------------------------------------------------------
bool SimpleTokenizer::init(const std::vector<std::string>& vocab, const std::vector<std::string>& merges) {
    if (vocab.empty() || merges.empty()) {
        GKD_LOG_ERROR("tokenizer: empty vocab or merges");
        return false;
    }
    for (size_t i = 0; i < vocab.size(); i++) {
        encoder_[vocab[i]] = (int32_t)i;
    }
    for (size_t r = 0; r < merges.size(); r++) {
        const std::string& m = merges[r];
        size_t sp = m.find(' ');
        if (sp == std::string::npos) continue;
        bpe_ranks_[m.substr(0, sp) + '\x01' + m.substr(sp + 1)] = (int32_t)r;
    }
    sot_id_ = encoder_["<|startoftext|>"];
    eot_id_ = encoder_["<|endoftext|>"];
    if (sot_id_ < 0 || eot_id_ < 0) {
        GKD_LOG_ERROR("tokenizer: SOT/EOT tokens missing from vocab");
        return false;
    }
    // bytes_to_unicode()
    std::vector<int> bs, cs;
    for (int b = '!'; b <= '~'; b++) bs.push_back(b);
    for (int b = 0xA1; b <= 0xAC; b++) bs.push_back(b);
    for (int b = 0xAE; b <= 0xFF; b++) bs.push_back(b);
    cs = bs;
    int n = 0;
    for (int b = 0; b < 256; b++) {
        if (std::find(bs.begin(), bs.end(), b) == bs.end()) {
            bs.push_back(b);
            cs.push_back(256 + n);
            n++;
        }
    }
    for (size_t i = 0; i < bs.size(); i++) {
        byte_encoder_[bs[i]].clear();
        utf8_append(byte_encoder_[bs[i]], (uint32_t)cs[i]);
    }
    return true;
}

std::vector<std::string> SimpleTokenizer::bpe(const std::string& token) const {
    auto it = cache_.find(token);
    if (it != cache_.end()) return it->second;

    // word = tuple(token[:-1]) + (token[-1] + '</w>',)
    // Symbols are unicode characters of the byte-encoded token string.
    std::vector<std::string> word;
    {
        size_t i = 0;
        std::vector<std::string> chars;
        while (i < token.size()) {
            size_t start = i;
            utf8_next(token, i);
            chars.push_back(token.substr(start, i - start));
        }
        if (chars.empty()) return {};
        for (size_t k = 0; k + 1 < chars.size(); k++) word.push_back(chars[k]);
        word.push_back(chars.back() + "</w>");
    }
    if (word.size() == 1) {
        cache_[token] = word;
        return word;
    }

    auto get_rank = [&](const std::string& a, const std::string& b) -> int {
        auto r = bpe_ranks_.find(a + '\x01' + b);
        return r == bpe_ranks_.end() ? INT32_MAX : r->second;
    };

    while (true) {
        // bigram = min(pairs, key=rank)
        int best_rank = INT32_MAX;
        size_t best_i = SIZE_MAX;
        for (size_t i = 0; i + 1 < word.size(); i++) {
            int r = get_rank(word[i], word[i + 1]);
            if (r < best_rank) {
                best_rank = r;
                best_i = i;
            }
        }
        if (best_i == SIZE_MAX || best_rank == INT32_MAX) break;
        const std::string first = word[best_i];
        const std::string second = word[best_i + 1];

        std::vector<std::string> new_word;
        size_t i = 0;
        while (i < word.size()) {
            size_t j = i;
            while (j < word.size() && word[j] != first) j++;
            new_word.insert(new_word.end(), word.begin() + i, word.begin() + j);
            i = j;
            if (i >= word.size()) break;
            if (word[i] == first && i < word.size() - 1 && word[i + 1] == second) {
                new_word.push_back(first + second);
                i += 2;
            } else {
                new_word.push_back(word[i]);
                i += 1;
            }
        }
        word = std::move(new_word);
        if (word.size() == 1) break;
    }

    cache_[token] = word;
    return word;
}

std::vector<int32_t> SimpleTokenizer::encode(const std::string& text_raw) const {
    // text = whitespace_clean(basic_clean(text)).lower()
    std::string text = text_raw;
    html_unescape(text);
    // strip + collapse whitespace
    std::string collapsed;
    bool in_space = false;
    size_t start = 0, end = text.size();
    while (start < end && (unsigned char)text[start] <= ' ') start++;
    while (end > start && (unsigned char)text[end - 1] <= ' ') end--;
    for (size_t i = start; i < end; i++) {
        unsigned char c = (unsigned char)text[i];
        if (c <= ' ') {
            if (!in_space) collapsed += ' ';
            in_space = true;
        } else {
            collapsed += text[i];
            in_space = false;
        }
    }
    lower_ascii(collapsed);

    // regex findall + byte-encode + BPE
    std::vector<int32_t> out;
    size_t i = 0;
    std::string byte_seq;  // byte-encoded current token
    auto emit = [&](const std::string& tok) {
        if (tok.empty()) return;
        byte_seq.clear();
        for (unsigned char b : tok) byte_seq += byte_encoder_[b];
        for (auto& sym : bpe(byte_seq)) {
            auto e = encoder_.find(sym);
            if (e != encoder_.end()) out.push_back(e->second);
        }
    };

    while (i < collapsed.size()) {
        size_t start = i;
        uint32_t cp = utf8_next(collapsed, i);
        Cat cat = category(cp);

        // contractions (case-insensitive, applied on lowercased text)
        if (cp == '\'' && start + 1 < collapsed.size()) {
            static const char* contr[] = {"'s", "'t", "'re", "'ve", "'m", "'ll", "'d"};
            bool matched = false;
            for (const char* c : contr) {
                size_t len = std::strlen(c);
                if (collapsed.compare(start, len, c) == 0) {
                    emit(collapsed.substr(start, len));
                    i = start + len;
                    matched = true;
                    break;
                }
            }
            if (matched) continue;
        }
        if (cat == Cat::Letter) {
            while (i < collapsed.size()) {
                size_t j = i;
                if (category(utf8_next(collapsed, j)) != Cat::Letter) break;
                i = j;
            }
            emit(collapsed.substr(start, i - start));
            continue;
        }
        if (cat == Cat::Digit) {
            // [\p{N}] matches exactly ONE numeric character
            emit(collapsed.substr(start, i - start));
            continue;
        }
        if (cat == Cat::Other) {
            while (i < collapsed.size()) {
                size_t j = i;
                Cat c2 = category(utf8_next(collapsed, j));
                if (c2 == Cat::Letter || c2 == Cat::Digit || c2 == Cat::Space) break;
                i = j;
            }
            emit(collapsed.substr(start, i - start));
            continue;
        }
        // Space: skipped by the regex
    }
    return out;
}

std::vector<int32_t> SimpleTokenizer::tokenize(const std::string& text, int context_length) const {
    std::vector<int32_t> tokens;
    tokens.push_back(sot_id_);
    std::vector<int32_t> ids = encode(text);
    tokens.insert(tokens.end(), ids.begin(), ids.end());
    tokens.push_back(eot_id_);
    if ((int)tokens.size() > context_length) {
        tokens.resize(context_length);
        tokens.back() = eot_id_;
    }
    std::vector<int32_t> result(context_length, 0);
    for (size_t k = 0; k < tokens.size() && k < (size_t)context_length; k++) result[k] = tokens[k];
    return result;
}

}  // namespace gkd
