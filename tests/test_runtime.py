from ptxbench.runtime import parse_ptxas_output


def test_parse_ptxas_output_extracts_verbose_resource_usage() -> None:
    output = """
ptxas info    : 0 bytes gmem
ptxas info    : Compiling entry function 'relu_kernel' for 'sm_89'
ptxas info    : Function properties for relu_kernel
    16 bytes stack frame, 8 bytes spill stores, 12 bytes spill loads
ptxas info    : Used 32 registers, used 2 barriers, 48 bytes smem, 24 bytes lmem, 352 bytes cmem[0], 16 bytes cmem[2]
ptxas info    : Compile time = 0.012 ms
"""

    report = parse_ptxas_output(output, source_name="relu", arch="sm_89")

    assert report.source_name == "relu"
    assert report.arch == "sm_89"
    assert report.registers == 32
    assert report.spill_stores_bytes == 8
    assert report.spill_loads_bytes == 12
    assert report.shared_memory_bytes == 48
    assert report.local_memory_bytes == 24
    assert report.constant_memory_bytes == 368
    assert report.stack_frame_bytes == 16
    assert len(report.functions) == 1
    assert report.functions[0].name == "relu_kernel"

