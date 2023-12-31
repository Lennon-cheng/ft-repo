# Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 3.8)

set(parallel_gpt_triton_backend_files
    Baichuan2TritonModel.cc
    Baichuan2TritonModelInstance.cc
)

add_library(Baichuan2TritonBackend STATIC ${parallel_gpt_triton_backend_files})
set_property(TARGET Baichuan2TritonBackend PROPERTY POSITION_INDEPENDENT_CODE  ON)
target_link_libraries(Baichuan2TritonBackend PRIVATE TransformerTritonBackend Baichuan2 tensor memory_utils -lcublasLt)
target_compile_features(Baichuan2TritonBackend PRIVATE cxx_std_14)
