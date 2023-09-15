# Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
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

add_library(Llama16kDecoderLayerWeight STATIC Llama16kDecoderLayerWeight.cc)
set_property(TARGET Llama16kDecoderLayerWeight PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET Llama16kDecoderLayerWeight PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(Llama16kDecoderLayerWeight PUBLIC memory_utils cuda_utils logger)

add_library(Llama16kDecoder STATIC Llama16kDecoder.cc)
set_property(TARGET Llama16kDecoder PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET Llama16kDecoder PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(Llama16kDecoder PUBLIC -lcudart cublasMMWrapper
                      TensorParallelDecoderSelfAttentionLayer
                      TensorParallelSiluFfnLayer
                      layernorm_kernels
                      add_residual_kernels
                      Llama16kDecoderLayerWeight
                      tensor
                      nccl_utils
                      cuda_utils
                      logger)

add_library(Llama16kContextDecoder STATIC Llama16kContextDecoder.cc)
set_property(TARGET Llama16kContextDecoder PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET Llama16kContextDecoder PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(Llama16kContextDecoder PUBLIC -lcudart cublasMMWrapper
                      TensorParallelGptContextAttentionLayer
                      TensorParallelSiluFfnLayer
                      layernorm_kernels
                      add_residual_kernels
                      gpt_kernels
                      tensor
                      nccl_utils
                      cuda_utils
                      logger)

add_library(Llama16kWeight STATIC Llama16kWeight.cc)
set_property(TARGET Llama16kWeight PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET Llama16kWeight PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(Llama16kWeight PUBLIC Llama16kDecoderLayerWeight cuda_utils logger)

add_library(Llama16k STATIC Llama16k.cc)
set_property(TARGET Llama16k PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET Llama16k PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(Llama16k PUBLIC -lcudart
                      Llama16kDecoder
                      Llama16kContextDecoder
                      decoding_kernels
                      gpt_kernels
                      DynamicDecodeLayer
                      BaseBeamSearchLayer
                      bert_preprocess_kernels
                      tensor
                      Llama16kWeight
                      cuda_utils
                      logger)
