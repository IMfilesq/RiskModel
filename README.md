# instalacja menadżera pakietów i środowiska
pip install uv
# sychronizacja
uv sync
# dodanie kucza api do pobierania danych
zaloguj się na https://www.tiingo.com/ zakładka account, token
utwórz w folderze głownym projektu plik .env wpisz w nim TIINGO_API_KEY=TwójToken
# odpalenie programu z domyślnymi wartościami konfiguracji
uv run riskmodel
# odpalenie programu z wartościami konfiguracji podmienionymi w locie
uv run riskmodel model.lr=0.001
# instalacja automatycznej weryfikacji kodu
uv run pre-commit install
