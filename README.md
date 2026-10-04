# instalacja menadżera pakietów i środowiska
pip install uv
# sychronizacja
uv sync
# odpalenie programu z domyślnymi wartościami konfiguracji
uv run python -m src.main
# odpalenie programu z wartościami konfiguracji podmienionymi w locie
uv run python -m src.main model.lr=0.001
# instalacja automatycznej weryfikacji kodu
uv run pre-commit install
