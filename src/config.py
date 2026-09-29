from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "cs-training.csv"

TARGET = "SeriousDlqin2yrs"

RANDOM_STATE = 42
TEST_SIZE = 0.2
N_SPLITS = 5


LATE_COLS = [
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfTimes90DaysLate",
]

# значения 96 и 98 в счётчиках — служебные коды
SENTINEL_MIN = 90

# условные единицы на клиента: суммы кредита в данных нет
GAIN = 1.0   # прибыль с одобренного добросовестного заёмщика
LOSS = -5.0  # убыток с одобренного заёмщика, ушедшего в дефолт
