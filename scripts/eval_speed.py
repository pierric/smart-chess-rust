#    eager: 1.36 ms
#    triton: 4.638 ms
#    traced-amp: 0.892 ms


import torch
import nn

#
# hidet doesn't work for now
# 
# import hidet
# hidet.torch.dynamo_config.search_space(2)
# model_opt = torch.compile(model, backend='hidet')

x = torch.randn(1, 119, 8, 8, dtype=torch.float32).cuda()
model = nn.ChessModule()
model = model.cuda().eval()

with torch.no_grad():
    model_triton = torch.compile(model, mode="reduce-overhead")

with torch.no_grad():
    with torch.autocast(device_type="cuda", dtype=torch.float16, cache_enabled=False):
        # import torch.fx.experimental.optimization as optimization
        # model_ = optimization.fuse(model)
        model_ = model
        model_jit = torch.jit.trace(model_, [x])
        model_jit = torch.jit.freeze(model_jit)

# run the optimized model
y1 = model_jit(x)
y2 = model_triton(x)
y3 = model(x)

print(y1[0].dtype, y2[0].dtype, y3[0].dtype)
print(y1[1].dtype, y2[1].dtype, y3[1].dtype)

# check the correctness
torch.testing.assert_close(actual=y1[0], expected=y2[0], rtol=1e-2, atol=1e-2)
torch.testing.assert_close(actual=y1[0], expected=y3[0], rtol=1e-2, atol=1e-2)
torch.testing.assert_close(actual=y1[1].float(), expected=y2[1], rtol=1e-2, atol=1e-2)
torch.testing.assert_close(actual=y1[1].float(), expected=y3[1], rtol=1e-2, atol=1e-2)


# benchmark the performance
for name, model in [
    ('eager', model),
    ('triton', model_triton),
    ('traced-amp', model_jit)
    ]:
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start_event.record()
    for _ in range(100):
        y = model(x)
    end_event.record()
    torch.cuda.synchronize()
    print('{:>10}: {:.3f} ms'.format(name, start_event.elapsed_time(end_event) / 100.0))
