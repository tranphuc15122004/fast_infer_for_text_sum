/*
 * Offline compatibility subset for the SSSD native extension.
 *
 * SSSD uses spdlog for diagnostic messages only.  The benchmark server cannot
 * clone third-party repositories, so this header implements the small API
 * surface used by SSSD while preserving the logger types and macros expected
 * by the vendored source.  A full local spdlog checkout can still be selected
 * with -DSSSD_SPDLOG_SOURCE_DIR=....
 */
#ifndef SSSD_SPDLOG_COMPAT_SPDLOG_H
#define SSSD_SPDLOG_COMPAT_SPDLOG_H

#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>

#ifndef SPDLOG_LEVEL_TRACE
#define SPDLOG_LEVEL_TRACE 0
#define SPDLOG_LEVEL_DEBUG 1
#define SPDLOG_LEVEL_INFO 2
#define SPDLOG_LEVEL_WARN 3
#define SPDLOG_LEVEL_ERROR 4
#define SPDLOG_LEVEL_CRITICAL 5
#define SPDLOG_LEVEL_OFF 6
#endif

namespace spdlog {

class logger;

namespace level {
enum level_enum {
    trace = SPDLOG_LEVEL_TRACE,
    debug = SPDLOG_LEVEL_DEBUG,
    info = SPDLOG_LEVEL_INFO,
    warn = SPDLOG_LEVEL_WARN,
    err = SPDLOG_LEVEL_ERROR,
    critical = SPDLOG_LEVEL_CRITICAL,
    off = SPDLOG_LEVEL_OFF,
};
}  // namespace level

namespace detail {

inline std::string format_message(const std::string &format)
{
    return format;
}

template <typename Value>
std::string stringify(Value &&value)
{
    std::ostringstream stream;
    stream << std::forward<Value>(value);
    return stream.str();
}

inline void append_formatted(std::ostringstream &out, const std::string &format)
{
    out << format;
}

template <typename Value, typename... Values>
void append_formatted(
    std::ostringstream &out, const std::string &format, Value &&value, Values &&...values)
{
    const std::size_t placeholder = format.find("{}");
    if (placeholder == std::string::npos) {
        out << format;
        return;
    }
    out << format.substr(0, placeholder);
    out << stringify(std::forward<Value>(value));
    append_formatted(
        out,
        format.substr(placeholder + 2),
        std::forward<Values>(values)...);
}

template <typename... Values>
std::string format(const std::string &format_string, Values &&...values)
{
    std::ostringstream out;
    append_formatted(out, format_string, std::forward<Values>(values)...);
    return out.str();
}

inline const char *level_name(level::level_enum value)
{
    switch (value) {
    case level::trace:
        return "trace";
    case level::debug:
        return "debug";
    case level::info:
        return "info";
    case level::warn:
        return "warning";
    case level::err:
        return "error";
    case level::critical:
        return "critical";
    case level::off:
        return "off";
    }
    return "unknown";
}

inline std::unordered_map<std::string, std::weak_ptr<logger>> &registry()
{
    static std::unordered_map<std::string, std::weak_ptr<logger>> value;
    return value;
}

inline std::mutex &registry_mutex()
{
    static std::mutex value;
    return value;
}

}  // namespace detail

class logger {
public:
    explicit logger(std::string name) : name_(std::move(name)) {}

    const std::string &name() const
    {
        return name_;
    }

    void set_pattern(const std::string &pattern)
    {
        pattern_ = pattern;
    }

    void set_level(level::level_enum value)
    {
        level_ = value;
    }

    template <typename... Values>
    void log(level::level_enum value, const std::string &format_string, Values &&...values)
    {
        if (value < level_ || level_ == level::off) {
            return;
        }
        std::lock_guard<std::mutex> lock(output_mutex());
        std::cerr << "[" << detail::level_name(value) << "] [" << name_ << "] "
                  << detail::format(format_string, std::forward<Values>(values)...)
                  << std::endl;
    }

    template <typename... Values>
    void trace(const std::string &format_string, Values &&...values)
    {
        log(level::trace, format_string, std::forward<Values>(values)...);
    }

    template <typename... Values>
    void debug(const std::string &format_string, Values &&...values)
    {
        log(level::debug, format_string, std::forward<Values>(values)...);
    }

    template <typename... Values>
    void info(const std::string &format_string, Values &&...values)
    {
        log(level::info, format_string, std::forward<Values>(values)...);
    }

    template <typename... Values>
    void warn(const std::string &format_string, Values &&...values)
    {
        log(level::warn, format_string, std::forward<Values>(values)...);
    }

    template <typename... Values>
    void error(const std::string &format_string, Values &&...values)
    {
        log(level::err, format_string, std::forward<Values>(values)...);
    }

private:
    static std::mutex &output_mutex()
    {
        static std::mutex value;
        return value;
    }

    std::string name_;
    std::string pattern_;
    level::level_enum level_ = level::info;
};

inline void register_logger(const std::shared_ptr<logger> &value)
{
    std::lock_guard<std::mutex> lock(detail::registry_mutex());
    detail::registry()[value->name()] = value;
}

inline std::shared_ptr<logger> get(const std::string &name)
{
    std::lock_guard<std::mutex> lock(detail::registry_mutex());
    const auto found = detail::registry().find(name);
    return found == detail::registry().end() ? nullptr : found->second.lock();
}

inline std::shared_ptr<logger> stdout_color_mt(const std::string &name)
{
    auto value = std::make_shared<logger>(name);
    register_logger(value);
    return value;
}

}  // namespace spdlog

#define SPDLOG_LOGGER_TRACE(logger, ...) \
    do {                                  \
        if (logger) {                     \
            (logger)->trace(__VA_ARGS__); \
        }                                     \
    } while (false)
#define SPDLOG_LOGGER_DEBUG(logger, ...) \
    do {                                  \
        if (logger) {                     \
            (logger)->debug(__VA_ARGS__); \
        }                                     \
    } while (false)
#define SPDLOG_LOGGER_INFO(logger, ...) \
    do {                                 \
        if (logger) {                    \
            (logger)->info(__VA_ARGS__); \
        }                                    \
    } while (false)
#define SPDLOG_LOGGER_WARN(logger, ...) \
    do {                                 \
        if (logger) {                    \
            (logger)->warn(__VA_ARGS__); \
        }                                    \
    } while (false)
#define SPDLOG_LOGGER_ERROR(logger, ...) \
    do {                                  \
        if (logger) {                     \
            (logger)->error(__VA_ARGS__); \
        }                                     \
    } while (false)

#endif  // SSSD_SPDLOG_COMPAT_SPDLOG_H
