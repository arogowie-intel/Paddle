/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <stdint.h>
#include <fstream>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include "paddle/fluid/framework/ddim.h"
#include "paddle/fluid/operators/save_op.h"
#include "paddle/fluid/platform/device_context.h"

namespace paddle {
namespace operators {
namespace detail {
namespace {

void allocateAndFillInt8Tensor(const framework::ExecutionContext& ctx,
                               const void* src,
                               const mkldnn::memory::desc& mem_desc,
                               framework::Tensor* tensor) {
  VLOG(2) << "Allocate and fill Tensor.";
  const mkldnn::memory::dims dims = mem_desc.dims();
  size_t bytes = mem_desc.get_size();
  int8_t* tensor_data_ptr = tensor->mutable_data<int8_t>(
      framework::make_ddim(dims), ctx.GetPlace(), bytes);
  std::memcpy(tensor_data_ptr, src, bytes);
}

const framework::Variable wrapInVariable(
    const framework::ExecutionContext& ctx,
    const std::shared_ptr<mkldnn::memory> mem_p) {
  const void* cached_data_ptr = mem_p->get_data_handle();
  const auto mem_desc = mem_p->get_desc();
  const auto data_type = mem_desc.data_type();
  framework::Variable var;

  if (data_type == mkldnn::memory::data_type::s8) {
    // Create empty tensor and allocate memory for it
    framework::Tensor* tensor = var.GetMutable<framework::Tensor>();
    allocateAndFillInt8Tensor(ctx, cached_data_ptr, mem_desc, tensor);
  }

  return var;
}

std::string getOneDNNCacheKey(const std::string& in_name) {
  std::string key_tid = "";
  if (platform::MKLDNNDeviceContext::tls().get_cur_mkldnn_session_id() ==
      platform::MKLDNNDeviceContextThreadLocals::kMKLDNNSessionID_Default) {
    key_tid = "-t:" + platform::ThreadIDasStr();
  }
  return platform::CreateKey(key_tid, in_name);
}

}  // namespace

const framework::Variable getOneDNNCachedVariable(
    const framework::ExecutionContext& ctx) {
  auto in_name = ctx.InputName("X");
  auto key = getOneDNNCacheKey(in_name);
  // TODO(aosewski): check whether ThreadID was used for cache keys
  auto& dev_ctx = ctx.template device_context<platform::MKLDNNDeviceContext>();
  const std::shared_ptr<mkldnn::memory> mem_p =
      std::static_pointer_cast<mkldnn::memory>(dev_ctx.GetBlob(key));
  if (!mem_p) {
    VLOG(2) << "Couldn't get variable: \"" << in_name
            << "\" from MKLDNN cache using \"" << key << "\" key";

    return framework::Variable();
  }
  VLOG(2) << "Retrieve from cache variable: " << in_name;
  return wrapInVariable(ctx, mem_p);
}

}  // namespace detail

class SaveOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {}

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    auto data_type = OperatorWithKernel::IndicateVarDataType(ctx, "X");
    return framework::OpKernelType(data_type, ctx.device_context());
  }
};

class SaveOpProtoMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor ) Input LoDTensor and SelectedRows to be saved");
    AddComment(R"DOC(
Save operator

This operator will serialize and write LoDTensor / SelectedRows variable to file on disk.
)DOC");
    AddAttr<bool>("overwrite",
                  "(boolean, default true)"
                  "Overwrite the output file if exist")
        .SetDefault(true);
    AddAttr<bool>("save_as_fp16",
                  "(boolean, default false)"
                  "If true, the tensor will be converted to float16 data "
                  "type and then saved. Otherwise, the tensor will be "
                  "directly saved without data type conversion.")
        .SetDefault(false);
    AddAttr<std::string>("file_path",
                         "(string)"
                         "The \"file_path\" where the variable will be saved.")
        .AddCustomChecker(
            [](const std::string& path) { return !path.empty(); });
    AddOutput(LOOKUP_TABLE_PATH,
              "(string)"
              "for pserver: The \"kLookupTablePath\" where checkpoint notify "
              "to save lookup table variables"
              " to directory specified.")
        .AsDispensable();
  }
};

class SaveOpVarTypeInference : public framework::VarTypeInference {
 public:
  void operator()(framework::InferVarTypeContext* ctx) const override {
    auto var_type = framework::proto::VarType::RAW;
    ctx->InsertVar(LOOKUP_TABLE_PATH, var_type);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(save, ops::SaveOp, ops::SaveOpProtoMaker,
                  ops::SaveOpVarTypeInference);

REGISTER_OP_CPU_KERNEL(
    save, ops::SaveOpKernel<paddle::platform::CPUDeviceContext, float>,
    ops::SaveOpKernel<paddle::platform::CPUDeviceContext, double>,
    ops::SaveOpKernel<paddle::platform::CPUDeviceContext, int>,
    ops::SaveOpKernel<paddle::platform::CPUDeviceContext, uint8_t>,
    ops::SaveOpKernel<paddle::platform::CPUDeviceContext, int8_t>,
    ops::SaveOpKernel<paddle::platform::CPUDeviceContext, int16_t>,
    ops::SaveOpKernel<paddle::platform::CPUDeviceContext, int64_t>);
