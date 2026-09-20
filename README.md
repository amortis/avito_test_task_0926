# avito_test_task_0926
# Классификация ботов — решение

## Подход

**Препроцессинг.** Нормализация регистра `platform` и `event_name`, замена `iphone → ios`,
удаление полных дублей с сохранением `duplicate_ratio`, сортировка по `(cookie_id, event_ts)`.

**Признаки (54 на куку).** Четыре группы:

1. **User-Agent / устройство**: браузер, ОС, версии, `is_headless`, `is_script`,
   `ver_gap` (насколько старая версия), `os_platform_mismatch`.
2. **Временные**: `n_events`, `span_sec`, статистики интервалов (`mean/median/std/min/max/cv`),
   `share_dt0`, `share_fast`, `n_hours`, `night_share`, `dt_top_share`, `events_per_hour`.
3. **Разнообразие и состав**: уникальные объявления/категории/локации/запросы,
   `page_max/mean`, `item_ratio`, `query_ratio`, доли типов событий, `duplicate_ratio`.
4. **Курсор** (только web): `ptr_share`, `ptr_n`, `ptr_nuniq`, `ptr_uniq_ratio`,
   `ptr_x_std`, `ptr_y_std`, `ptr_step_mean/std`, `ptr_step0_share`.

**Модель.** `HistGradientBoostingClassifier` с нативными категориальными признаками
(`categorical_features=[device, ua_type, browser, os]`).



**Результаты P@R0.7 на временных срезах:**

| cutoff | n_valid | pos | P@R0.7 |
|---|---|---|---|
| 2026-04-13 | 5161 | 408 | 0.696 |
| 2026-04-17 | 1951 | 160 | 0.752 |


**Воспроизводимость.** `random_state=0`.
Python 3.12, зависимости в `requirements.txt`.
