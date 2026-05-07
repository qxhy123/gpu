from __future__ import annotations
from pathlib import Path
import typer
import numpy as np

app = typer.Typer(help="gpusim — teaching-oriented GPU simulator")


def _parse_dim(s: str) -> tuple[int,int,int]:
    parts = [int(x) for x in s.split(",")]
    while len(parts) < 3: parts.append(1)
    return tuple(parts[:3])  # type: ignore[return-value]


def _parse_inputs(s: str | None) -> dict[str, str]:
    if not s: return {}
    out: dict[str, str] = {}
    for chunk in s.split(","):
        name, _, path = chunk.partition(":")
        out[name.strip()] = path.strip()
    return out


@app.command()
def run(
    kernel: Path,
    grid: str = typer.Option(..., "--grid"),
    block: str = typer.Option(..., "--block"),
    inputs: str = typer.Option(None, "--inputs"),
    config: Path = typer.Option(None, "--config"),
    output: Path = typer.Option(None, "--output"),
    perfetto: Path = typer.Option(None, "--perfetto"),
    trace: Path = typer.Option(None, "--trace"),
    mode: str = typer.Option("functional", "--mode"),
    seed: int = typer.Option(0, "--seed"),
):
    """Run a PTX kernel."""
    from gpusim.api import run as api_run
    g = _parse_dim(grid); b = _parse_dim(block)
    inps = _parse_inputs(inputs)
    params: dict[str, np.ndarray | int] = {}
    np_paths: dict[str, Path] = {}
    for name, path in inps.items():
        if path.endswith(".npy"):
            arr = np.load(path)
            params[name] = arr
            np_paths[name] = Path(path)
        else:
            # scalar int param
            params[name] = int(path)
    src = kernel.read_text()
    res = api_run(ptx_src=src, grid=g, block=b, params=params, mode=mode,
                  config=config, seed=seed)
    typer.echo(res.summary())
    # save back numpy arrays so caller can inspect them
    for name, p in np_paths.items():
        if name in res.outputs:
            np.save(p, res.outputs[name])
    if output:
        res.html_report(output)
    if perfetto:
        res.perfetto(perfetto)
    if trace and res._recorder is not None:
        from gpusim.trace.writer import write_parquet
        write_parquet(res._recorder, trace)


@app.command()
def explain(report: Path):
    """Grep cycles + bottleneck out of an HTML report."""
    text = report.read_text()
    import re
    m = re.search(r"Cycles</th><td>(\d+)</td>", text)
    if m:
        typer.echo(f"cycles: {m.group(1)}")
    m2 = re.search(r"Bottleneck</th><td>(\w+)</td>", text)
    if m2:
        typer.echo(f"bottleneck: {m2.group(1)}")


@app.command()
def show(kernel: Path):
    """Show parsed IR + IPDOM annotations."""
    from gpusim.frontend.parser import parse
    k = parse(kernel.read_text(), str(kernel))
    typer.echo(f"kernel: {k.name}")
    typer.echo(f"params: {[(p.name, p.type.value) for p in k.params]}")
    typer.echo(f"regs: s32={k.regs.s32} u32={k.regs.u32} u64={k.regs.u64} "
               f"f32={k.regs.f32} pred={k.regs.pred}")
    typer.echo(f"instrs: {len(k.instrs)}, labels: {list(k.labels)}")
    if k.ipdom:
        typer.echo(f"ipdom: {k.ipdom}")


@app.command()
def doctor():
    """Verify dependencies and report versions."""
    import numpy, pandas, pyarrow, plotly, jinja2, yaml
    typer.echo(f"numpy {numpy.__version__}")
    typer.echo(f"pandas {pandas.__version__}")
    typer.echo(f"pyarrow {pyarrow.__version__}")
    typer.echo(f"plotly {plotly.__version__}")
    typer.echo(f"jinja2 {jinja2.__version__}")
    typer.echo(f"pyyaml {yaml.__version__}")
    typer.echo("OK")
