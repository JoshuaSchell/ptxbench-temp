from ptxbench import runtime
from ptxbench.runtime import clear_ptx_artifact_log, get_ptx_artifact_log, parse_ptxas_output, summarize_ptx_artifact_resources


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


def test_ptx_resource_summary_aggregates_artifact_reports() -> None:
    report = parse_ptxas_output(
        """
ptxas info    : Compiling entry function 'a' for 'sm_89'
ptxas info    : Function properties for a
    0 bytes stack frame, 0 bytes spill stores, 4 bytes spill loads
ptxas info    : Used 20 registers, 8 bytes smem, 0 bytes lmem, 16 bytes cmem[0]
ptxas info    : Compiling entry function 'b' for 'sm_89'
ptxas info    : Function properties for b
    8 bytes stack frame, 16 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 44 registers, 32 bytes smem, 2 bytes lmem, 8 bytes cmem[0]
""",
        source_name="multi",
        arch="sm_89",
    )
    summary = summarize_ptx_artifact_resources([{"assembly_report": report.to_dict()}])

    assert summary["num_artifacts"] == 1
    assert summary["num_functions"] == 2
    assert summary["max_registers"] == 44
    assert summary["max_spill_stores_bytes"] == 16
    assert summary["max_spill_loads_bytes"] == 4
    assert summary["any_spills"] is True


def test_ptx_artifact_log_dedupes_by_source_hash_and_entry() -> None:
    clear_ptx_artifact_log()
    try:
        runtime._PTX_ARTIFACT_LOG.extend(
            [
                {"source_hash": "abc", "entry": "kernel", "assembly_report": {"functions": []}, "marker": 1},
                {"source_hash": "abc", "entry": "kernel", "assembly_report": {"functions": []}, "marker": 2},
                {"source_hash": "abc", "entry": "other", "assembly_report": {"functions": []}, "marker": 3},
            ]
        )
        records = get_ptx_artifact_log()
        assert len(records) == 2
        assert {(record["entry"], record["marker"]) for record in records} == {("kernel", 2), ("other", 3)}
    finally:
        clear_ptx_artifact_log()
