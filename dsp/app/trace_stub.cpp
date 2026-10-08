// SPDX-License-Identifier: Apache-2.0
// app/server/http.cpp logs through engine::debug, whose implementation (trace.cpp) pulls in
// ggml. audiocpp_dsp has no ggml, so it answers those two calls itself: logging off.
#include "engine/framework/debug/trace.h"

namespace engine::debug {

bool log_enabled() {
    return false;
}

void log_message(std::string_view) {}

}  // namespace engine::debug
