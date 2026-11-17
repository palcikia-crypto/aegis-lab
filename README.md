# AEGIS-Lab v3 PRO · Adaptive Fee-Edge Engine (Binance/OKX/Bybit, GitHub Actions)

Paper-версия «AEGIS-1:1 QUANT ENGINE» под бюджет $0: весь движок живёт в GitHub
Actions на публичных рыночных данных, без API-ключей и без реальных денег.
v3 PRO — система сама развивается: скачивает нужные монеты и их торговые
спецификации, ищет эдж по режимам рынка и защищает прибыль от комиссий.

## Архитектура v3 PRO
- **Данные**: Binance USDT-M futures (лучшие лимиты/надёжность) → Bybit → OKX,
  автоматический фолбэк. Ключи не нужны.
- **Авто-вселенная** (`src/universe.py`): система сама скачивает топ-ликвидные
  USDT-перпы (по объёму 24ч, включая золото XAUUSDT, когда оно в топе) и их
  торговые спецификации (шаг лота, мин. количество, мин. нотинал) —
  ничего вручную конфигурировать не нужно.
- **Fee-Edge Envelope** (защита от комиссий): сделка открывается только если
  целевое движение ≥ N× круговых издержек (комиссии + спред-буфер).
  Бэктест: тот же период, тот же движок — без конверта −$32, с конвертом
  (N=5) +$3.2 при WR 52%. Оптимизатор сам подбирает N.
- **Адаптивный RR и TTL**: TP = rr×D (rr из WFO), а TTL вычисляется из
  «физики» — времени, за которое цена проходит D при текущем ATR-темпе
  (с буфером). Убивает главную утечку — таймауты до касания TP/SL.
- **Режим рынка** (`src/regime.py`): ADX(15m) + перцентиль ATR →
  trend_up / trend_down / range / volatile.
- **Стратегии** (`src/strategies.py`): baseline (объём+EMA+моментум),
  mean_reversion (BB+RSI, только рейндж), breakout (Donchian 15m + объём,
  против тренда не торгует), trend_follow (ADX-подтверждённый моментум).
- **Адаптивный мозг**: ewma R-мультипликатора по каждой связке
  «стратегия|режим»; связки с отрицательным ожиданием отключаются сами,
  при деградации снова включаются. Сид — из песочницы (opt.json).
- **Оптимизатор WFO** (`src/optimize.py`): грид (ATR k × vol-порог),
  робастность = прибыльность в каждом временном чанке + PF + WR.
  Лучшие параметры воркер подхватывает автоматически.
- **ML-гейт** (`src/train.py`): логрегрессия на 8 признаках, WFO 70/30,
  Monte Carlo 1000 прогонов. Фильтрует слабые сигналы в живом цикле.

## Что внутри
- **Движок 1:1** (`src/engine.py`): риск фиксированный в USDT, TP/SL = ±D,
  D = k×ATR(15m), экспирация 300/900 сек, принудительное закрытие по таймеру.
- **Признаки** (`src/features.py`): EMA9/21, моментум, всплеск объёма, ATR15.
  Признаки пишутся в каждую сделку — датасет для будущего ML-фильтра (Этап 3).
- **Данные** (`src/market.py`): Bybit V5 public REST (bytick → bybit),
  фолбэк на OKX public candles. Ключи не нужны.
- **Предохранители**: 6 убытков подряд → пауза 60 мин; −5% от дневного
  баланса → стоп до 00:00 UTC; максимум 3 одновременных позиции.
- **Дашборд**: `docs/index.html` (GitHub Pages), перегенерируется каждый цикл.
- **Надёжность**: state.json и trades.jsonl коммитятся каждый прогон,
  краши пишутся в results/crash.log и валят воркфлоу (не глотаются).

## Запуск
Воркер запускается автоматически каждые 5 минут (`.github/workflows/aegis.yml`).
Локально: `python -m src.worker`.

## Дашборд
Включите GitHub Pages: Settings → Pages → Source: `main` / `/docs`.
Панель: https://palcikia-crypto.github.io/aegis-lab/

## Cloudflare Pages
Панель дополнительно деплоится на Cloudflare Pages (`.github/workflows/cloudflare-pages.yml`).

Требуемые секреты репозитория (Settings → Secrets and variables → Actions):
- `CLOUDFLARE_API_TOKEN` — API-токен Cloudflare (Account API Tokens)
- `CLOUDFLARE_ACCOUNT_ID` — Account ID
- опционально для R2: `CLOUDFLARE_R2_ACCESS_KEY_ID`, `CLOUDFLARE_R2_SECRET_ACCESS_KEY`

Повторное использование в любом другом репозитории (секреты должны быть заданы в нём же):

```yaml
jobs:
  deploy:
    uses: palcikia-crypto/aegis-lab/.github/workflows/cloudflare-pages.yml@main
    with:
      directory: docs
      project_name: my-project
    secrets: inherit
```

## Дорожная карта (по ТЗ)
1. ✅ Этап 1: базовый каркас + калькулятор риска + таймеры (этот репо, paper).
2. ✅ Этап 2 (частично): запись признаков каждой сделки → датасет.
3. ✅ Этап 3: ML-фильтр на собранных логах, WFO + Monte Carlo (pure-python
   логистическая регрессия, без зависимостей).
4. ⏳ Этап 4: реальный депозит — только после gold-проверки, ключи через Secrets.

## Песочница (ускоренное обучение)
Живой воркер набирает датасет неделями. Песочница (`src/sim.py`) решает это:
реплей истории 1m-свечей через тот же движок (те же комиссии, риск,
предохранители), но на исторических данных — 14 дней × 3 символа прогоняются
за ~минуту. Сделки пишутся в `results/sim_trades.jsonl` с префиксом `sim-` и
пометкой `source: "sim"`; живое paper-состояние не затрагивается.

```bash
python -m src.optimize --days 30  # WFO-грид параметров → opt.json
python -m src.sim --days 30       # реплей истории в песочницу
python -m src.train               # обучение ML-гейта (sim + live сделки)
python -m src.worker              # живой цикл
```

`src/train.py` (Этап 3): логистическая регрессия на признаках сделок,
walk-forward 70/30, Monte Carlo 1000 прогонов по PnL. Модель пишется в
`results/model.json`. Когда модель есть, живой воркер гейтит сигналы:
сигнал открывается только если вероятность победы ≥ порога из валидации.
Без модели поведение воркера не меняется. В Actions песочница + обучение
запускаются ежедневно (cron 02:25 UTC) и коммитят датасет с моделью.

**Paper-only.** Это исследовательский стенд, не инвестиционная рекомендация.
## Cloudflare Workers
`cloudflare/gemini-proxy/worker.js` — прокси Google Gemini API через Cloudflare (обход региональных блокировок).
Живой воркер: https://gemini-proxy.apigemenipp.workers.dev

Использование: заменить базовый URL `https://generativelanguage.googleapis.com` на адрес воркера,
API-ключ Gemini передаётся как обычно (`x-goog-api-key` или `?key=`). CORS включён — работает из браузера.

Пример:
```
GET https://gemini-proxy.apigemenipp.workers.dev/v1beta/models?key=ВАШ_GEMINI_KEY
POST https://gemini-proxy.apigemenipp.workers.dev/v1beta/models/gemini-2.0-flash:generateContent?key=ВАШ_GEMINI_KEY
```

Деплой из репозитория (локально): `npx wrangler@3 deploy cloudflare/gemini-proxy/worker.js --name gemini-proxy --compatibility-date 2024-12-01`
