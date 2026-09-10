#include <cstddef>
#include <memory>
#include <new>

#include <omp.h>
#include <oneapi/dnnl/dnnl.hpp>

// This narrow C ABI deliberately owns only a prepared, F32 3x3 convolution.
// Model formats and tensor allocation remain Rust concerns in vestra-engine.
struct vestra_onednn_conv2d {
    dnnl::engine engine;
    dnnl::stream stream;
    dnnl::convolution_forward primitive;
    dnnl::memory weights;
    dnnl::memory bias;
    dnnl::memory scratchpad;
    dnnl::memory::desc src_desc;
    dnnl::memory::desc dst_desc;

    vestra_onednn_conv2d(dnnl::engine e, dnnl::stream s,
            dnnl::convolution_forward p, dnnl::memory w, dnnl::memory b,
            dnnl::memory scratch, dnnl::memory::desc src,
            dnnl::memory::desc dst)
        : engine(e), stream(s), primitive(p), weights(w), bias(b),
          scratchpad(scratch), src_desc(src), dst_desc(dst) {}
};

extern "C" vestra_onednn_conv2d *vestra_onednn_conv2d_create(
        std::size_t in_channels, std::size_t out_channels, std::size_t height,
        std::size_t width, const float *weight, const float *bias) {
    if (in_channels == 0 || out_channels == 0 || height == 0 || width == 0
            || weight == nullptr || bias == nullptr) {
        return nullptr;
    }

    try {
        // The host owns the global 16-thread budget. Reassert it here before
        // oneDNN creates its primitive/JIT state, and disable dynamic teams.
        omp_set_dynamic(0);
        omp_set_num_threads(16);

        const auto engine = dnnl::engine(dnnl::engine::kind::cpu, 0);
        const auto stream = dnnl::stream(engine);
        const auto src_desc = dnnl::memory::desc(
                {1, static_cast<dnnl::memory::dim>(in_channels),
                        static_cast<dnnl::memory::dim>(height),
                        static_cast<dnnl::memory::dim>(width)},
                dnnl::memory::data_type::f32, dnnl::memory::format_tag::nchw);
        const auto dst_desc = dnnl::memory::desc(
                {1, static_cast<dnnl::memory::dim>(out_channels),
                        static_cast<dnnl::memory::dim>(height),
                        static_cast<dnnl::memory::dim>(width)},
                dnnl::memory::data_type::f32, dnnl::memory::format_tag::nchw);
        const auto weights_desc = dnnl::memory::desc(
                {static_cast<dnnl::memory::dim>(out_channels),
                        static_cast<dnnl::memory::dim>(in_channels), 3, 3},
                dnnl::memory::data_type::f32, dnnl::memory::format_tag::oihw);
        const auto bias_desc = dnnl::memory::desc(
                {static_cast<dnnl::memory::dim>(out_channels)},
                dnnl::memory::data_type::f32, dnnl::memory::format_tag::x);
        const auto desc = dnnl::convolution_forward::desc(
                dnnl::prop_kind::forward_inference,
                dnnl::algorithm::convolution_direct, src_desc, weights_desc,
                bias_desc, dst_desc, {1, 1}, {1, 1}, {1, 1});
        dnnl::primitive_attr attr;
        attr.set_fpmath_mode(dnnl::fpmath_mode::strict);
        attr.set_accumulation_mode(dnnl::accumulation_mode::strict);
        attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
        const auto pd = dnnl::convolution_forward::primitive_desc(desc, attr, engine);
        const auto primitive = dnnl::convolution_forward(pd);

        const auto weight_source = dnnl::memory(weights_desc, engine,
                const_cast<float *>(weight));
        const auto prepared_weights = dnnl::memory(pd.weights_desc(), engine);
        dnnl::reorder(weight_source, prepared_weights).execute(stream, weight_source,
                prepared_weights);
        const auto prepared_bias = dnnl::memory(bias_desc, engine,
                const_cast<float *>(bias));
        const auto scratchpad = dnnl::memory(pd.scratchpad_desc(), engine);
        stream.wait();
        return new vestra_onednn_conv2d(engine, stream, primitive,
                prepared_weights, prepared_bias, scratchpad, src_desc, dst_desc);
    } catch (...) {
        return nullptr;
    }
}

extern "C" int vestra_onednn_conv2d_execute(vestra_onednn_conv2d *prepared,
        const float *input, float *output) {
    if (prepared == nullptr || input == nullptr || output == nullptr) return 0;
    try {
        omp_set_dynamic(0);
        omp_set_num_threads(16);
        const auto src = dnnl::memory(prepared->src_desc, prepared->engine,
                const_cast<float *>(input));
        const auto dst = dnnl::memory(prepared->dst_desc, prepared->engine, output);
        prepared->primitive.execute(prepared->stream,
                {{DNNL_ARG_SRC, src}, {DNNL_ARG_WEIGHTS, prepared->weights},
                        {DNNL_ARG_BIAS, prepared->bias}, {DNNL_ARG_DST, dst},
                        {DNNL_ARG_SCRATCHPAD, prepared->scratchpad}});
        prepared->stream.wait();
        return 1;
    } catch (...) {
        return 0;
    }
}

extern "C" void vestra_onednn_conv2d_destroy(vestra_onednn_conv2d *prepared) {
    delete prepared;
}
