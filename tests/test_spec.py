from ptxbench.spec import PTXKernelSpec


def test_kernel_spec_resolves_callable_dims() -> None:
    spec = PTXKernelSpec(
        entry="kernel",
        grid=lambda x: (x, 1, 1),
        block=(256, 1, 1),
        arg_types=("uint32",),
    )
    assert spec.resolve_grid(4) == (4, 1, 1)
    assert spec.resolve_block(4) == (256, 1, 1)
