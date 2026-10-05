# instalacja menadżera pakietów i środowiska
pip install uv
# sychronizacja
uv sync
# dodanie kucza api do pobierania danych
zaloguj się na https://www.tiingo.com/ zakładka account, token
utwórz w folderze głownym plik .env wpisz w nim TIINGO_API_KEY=TwójToken
# odpalenie programu z domyślnymi wartościami konfiguracji
uv run python -m src.main
# odpalenie programu z wartościami konfiguracji podmienionymi w locie
uv run python -m src.main model.lr=0.001
# instalacja automatycznej weryfikacji kodu
uv run pre-commit install
