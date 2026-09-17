# P3 — Confidence calibration from measured accuracy — design

Дата: 2026-09-17
Статус: чернетка на затвердження
Частина: P3 (залишок — per-record таблиця вже зроблена в A3). Мета — **чесна confidence**, а не сигнал.

## Мета
Зробити число confidence осмисленим: темперувати його **виміряним історичним hit-rate** класу
вердикту з backtest. **Ніколи не міняти вердикт/score** — лише знижувати впевненість, коли система
історично помиляється на цьому класі. Прозоро (примітка на VerdictCard).

## Чому саме так (з урахуванням виміру)
IC від'ємний, hit sell ≈ 14–33%. Калібрувати НАПРЯМ було б хибно (гнатися за шумом). Єдина чесна
калібрація — **знизити фальшиву впевненість**: якщо клас історично гірший за монетку, confidence
має це відображати. Це про **довіру**, не про edge.

## Формула (тільки знижує)
`factor = min(1.0, 2 * hit_rate[class@horizon])`, `calibrated = confidence * factor`.
- Клас із hit ≥ 50% → factor 1.0 (не чіпаємо).
- Клас із hit < 50% → темперуємо (sell 14% → ×0.28; sell 33% → ×0.66).
- `hold` (не напрямний, нема hit) і відсутній клас → factor 1.0.
Вердикт і score — **без змін**.

## Дизайн

### `eval/calibration.py`
- `build_calibration(records, horizons) -> dict`: `{str(h): hit_rate_by_class(rows@h)}` (переюз
  наявних метрик).
- `calibration_factor(calib, verdict, horizon) -> float`: за формулою вище; безпечні дефолти (1.0).

### Артефакт `calibration.json`
- Генерується з A3-backtest (я зберу його з наявного звіту 60 записів — без нового довгого прогону).
- Формат: `{"21": {"buy": 0.62, "sell": 0.33}, "63": {...}}`.

### Config
- `calibration_enabled: bool = False`, `calibration_path: str = "calibration.json"`,
  `calibration_horizon: int = 21`.

### Aggregator
- `Aggregator.__init__(..., calibration=None)`. Коли `calibration` передано і в config увімкнено:
  після risk-гейта `confidence *= calibration_factor(calibration, verdict, cfg.calibration_horizon)`
  і встановити `calibration_note`.
- `Verdict.calibration_note: str | None = None` (напр. «confidence tempered ×0.28 by historical sell accuracy»).

### CLI
- `analyze_ticker`/`build_backtest_verdict`: коли `cfg.calibration_enabled` і файл існує — завантажити
  `calibration.json`, передати в `Aggregator`. Немає файлу → без калібрації (factor 1.0, тихо).
- `config.yaml`: `calibration_enabled: true`.

### Frontend (мінімально)
- `Verdict.calibration_note?: string` у типах; VerdictCard показує дрібну примітку під confidence-метром.

## Файли
- `equity_research/eval/calibration.py` (новий), `equity_research/orchestration/aggregator.py`,
  `equity_research/config.py`, `equity_research/cli.py`, `config.yaml`, `calibration.json` (артефакт).
- Frontend: `src/api/types.ts`, `src/components/VerdictCard.tsx`.
- Тести: `test_calibration.py` (build + factor), `test_aggregator.py` (темперування + no-op коли ≥50%),
  Vitest на VerdictCard-примітку.

## Тестування
- `build_calibration` на фейкових records; `calibration_factor` (sell<0.5 темперує, buy≥0.5 =1.0,
  hold=1.0, відсутнє=1.0).
- Aggregator: з calibration — confidence темперовано для sell, не чіпнуто для buy; вердикт/score
  **не змінені**; без calibration — як зараз (усі наявні тести зелені).
- Жива: згенерувати `calibration.json`; analyze — sell-вердикти показують нижчу confidence + примітку;
  вердикти незмінні. Golden лишається (вердикт/стенси не міняються).

## Чесна засторога
Це робить confidence чеснішим (часто — нижчим), а не точнішим у передбаченні. Калібрація тягнеться з
одного бичачого періоду (60 записів) — орієнтир, не істина; онови `calibration.json` при новому вимірі.

## Відкриті рішення (підтвердити)
1. Формула `min(1, 2×hit)` — тільки знижує, не чіпає класи ≥50% (реком.) — ок?
2. Джерело — `calibration.json` з наявного A3-backtest, горизонт 21д (реком.) — ок?

## Поза межами
Калібрація напряму/score (хибно при IC<0); A1; сигнальні зміни.
