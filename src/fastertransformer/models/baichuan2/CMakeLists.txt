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

add_library(Baichuan2DecoderLayerWeight STATIC Baichuan2DecoderLayerWeight.cc)
set_property(TARGET Baichuan2DecoderLayerWeight PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET Baichuan2DecoderLayerWeight PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(Baichuan2DecoderLayerWeight PUBLIC memory_utils cuda_utils logger)

add_library(Baichuan2Decoder STATIC Baichuan2Decoder.cc)
set_property(TARGET Baichuan2Decoder PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET Baichuan2Decoder PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(Baichuan2Decoder PUBLIC -lcudart cublasMMWrapper
                      TensorParallelDecoderSelfAttentionLayer
                      TensorParallelSiluFfnLayer
                      layernorm_kernels
                      add_residual_kernels
                      Baichuan2DecoderLayerWeight
                      tensor
                      nccl_utils
                      cuda_utils
                      logger)

add_library(Baichuan2ContextDecoder STATIC Baichuan2ContextDecoder.cc)
set_property(TARGET Baichuan2ContextDecoder PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET Baichuan2ContextDecoder PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(Baichuan2ContextDecoder PUBLIC -lcudart cublasMMWrapper
                      TensorParallelGptContextAttentionLayer
                      TensorParallelSiluFfnLayer
                      layernorm_kernels
                      add_residual_kernels
                      gpt_kernels
                      tensor
                      nccl_utils
                      cuda_utils
                      logger)

add_library(Baichuan2Weight STATIC Baichuan2Weight.cc)
set_property(TARGET Baichuan2Weight PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET Baichuan2Weight PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(Baichuan2Weight PUBLIC Baichuan2DecoderLayerWeight cuda_utils logger)

add_library(Baichuan2 STATIC Baichuan2.cc)
set_property(TARGET Baichuan2 PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET Baichuan2 PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(Baichuan2 PUBLIC -lcudart
                      Baichuan2Decoder
                      Baichuan2ContextDecoder
                      decoding_kernels
                      gpt_kernels
                      DynamicDecodeLayer
                      BaseBeamSearchLayer
                      bert_preprocess_kernels
                      tensor
                      Baichuan2Weight
                      cuda_utils
                      logger)
