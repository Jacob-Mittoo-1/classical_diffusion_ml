def test_import() -> None:
    try:
        import classical_diffusion_ml  # noqa: PLC0415
    except ImportError:
        classical_diffusion_ml = None  # ty:ignore[invalid-assignment]

    assert classical_diffusion_ml is not None, (
        "classical_diffusion_ml module should not be None"
    )
