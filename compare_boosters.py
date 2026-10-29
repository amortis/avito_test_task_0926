"""Сравнение HGB с LightGBM и CatBoost на той же схеме валидации, что и в solution.ipynb.

Запуск из корня проекта:
    pip install lightgbm catboost
    python compare_boosters.py

Признаки берутся из solution.ipynb: скрипт выполняет его ячейки до определения make_model()
включительно, поэтому таблица признаков и эталонная HGB всегда совпадают с ноутбуком.

Протокол (тот же, что в ноутбуке, раздел «Подбор гиперпараметров»):
  * tune  — дни 04-12..04-15: каждый день предсказывается моделью, обученной на всех
            предыдущих днях; по этой метрике выбираются гиперпараметры;
  * hold  — дни 04-16..04-19, та же схема; смотрится только для финалистов.
"""
import contextlib
import io
import json
import time
import warnings

import numpy as np
from sklearn.model_selection import ParameterSampler

warnings.filterwarnings('ignore')

# --- 1. Признаки из ноутбука ---------------------------------------------------------
nb = json.load(open('solution.ipynb', encoding='utf-8'))
ns = {'__name__': 'solution'}
for cell in nb['cells']:
    if cell['cell_type'] != 'code':
        continue
    code = ''.join(cell['source'])
    with contextlib.redirect_stdout(io.StringIO()):      # выводы ноутбука здесь не нужны
        exec(compile(code, 'solution.ipynb', 'exec'), ns)
    if 'def make_model' in code:
        break
print('признаки готовы:', ns['Xtr_enc'].shape, flush=True)

X, y, train = ns['Xtr_enc'], ns['ytr'], ns['train']
cat_cols, cat_idx = ns['cat_cols'], ns['cat_idx']
precision_at_recall = ns['precision_at_recall']
HistGradientBoostingClassifier = ns['HistGradientBoostingClassifier']

d = train.window_start_ts.dt.normalize().values
days = np.sort(np.unique(d))
TUNE, HOLD = days[-8:-4], days[-4:]


def rolling(make, ds, Xm=X):
    ys, ps = [], []
    for day in ds:
        va, tr = d == day, d < day
        m = make().fit(Xm[tr], y[tr])
        ys.append(y[va])
        ps.append(m.predict_proba(Xm[va])[:, 1])
    return precision_at_recall(np.concatenate(ys), np.concatenate(ps))


def search(name, factory, space, n_iter, Xm=X):
    print(f'\n=== {name}: {n_iter} конфигураций ===', flush=True)
    res, t0 = [], time.time()
    for i, p in enumerate(ParameterSampler(space, n_iter=n_iter, random_state=0)):
        r = rolling(lambda: factory(**p), TUNE, Xm)
        res.append((r, p))
        print(f'{i:3d}  tune={r:.4f}  {p}  [{time.time() - t0:.0f}s]', flush=True)
    best_tune, best_p = max(res, key=lambda t: t[0])
    best_hold = rolling(lambda: factory(**best_p), HOLD, Xm)
    return {'model': name, 'tune': best_tune, 'hold': best_hold, 'params': best_p}


rows = []

# --- 2. HGB — эталон из ноутбука ---------------------------------------------------
make_model = ns['make_model']
rows.append({'model': 'HGB (как в ноутбуке)', 'tune': rolling(make_model, TUNE),
             'hold': rolling(make_model, HOLD), 'params': 'make_model()'})
print(rows[-1], flush=True)

# --- 3. LightGBM --------------------------------------------------------------------
try:
    import lightgbm as lgb

    X_lgb = X.copy()
    for c in cat_cols:                     # коды уже согласованы между train и test
        X_lgb[c] = X_lgb[c].astype('category')

    def lgb_factory(**p):
        return lgb.LGBMClassifier(subsample_freq=1, verbose=-1, random_state=0, n_jobs=-1, **p)

    lgb_space = {'n_estimators': [200, 400, 800, 1200],
                 'learning_rate': [0.02, 0.03, 0.05, 0.1],
                 'num_leaves': [7, 15, 31, 63],
                 'min_child_samples': [5, 10, 20, 40, 80],
                 'reg_lambda': [0, 0.1, 1, 3, 10],
                 'colsample_bytree': [0.5, 0.7, 1.0],
                 'subsample': [0.7, 0.85, 1.0],
                 'class_weight': [None, 'balanced']}
    rows.append(search('LightGBM', lgb_factory, lgb_space, n_iter=30, Xm=X_lgb))
except ImportError:
    print('lightgbm не установлен — пропускаю')

# --- 4. CatBoost --------------------------------------------------------------------
try:
    from catboost import CatBoostClassifier

    def cb_factory(**p):
        return CatBoostClassifier(cat_features=cat_cols, verbose=0, random_seed=0,
                                  thread_count=-1, **p)

    cb_space = {'iterations': [500, 1000, 1500],
                'learning_rate': [0.03, 0.05, 0.1],
                'depth': [4, 6, 8],
                'l2_leaf_reg': [1, 3, 10],
                'auto_class_weights': [None, 'Balanced']}
    rows.append(search('CatBoost', cb_factory, cb_space, n_iter=15))
except ImportError:
    print('catboost не установлен — пропускаю')

# --- 5. Итог -------------------------------------------------------------------------
print('\n' + '=' * 70)
print(f'{"модель":<24}{"tune 04-12..15":>16}{"hold 04-16..19":>16}')
for r in rows:
    print(f'{r["model"]:<24}{r["tune"]:>16.3f}{r["hold"]:>16.3f}')
print('\nЛучшие параметры:')
for r in rows:
    print(f'  {r["model"]}: {r["params"]}')
